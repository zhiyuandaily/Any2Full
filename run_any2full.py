# -*- coding: utf-8 -*-
"""
run_any2full.py — Any2Full inference for single RGBD or batch RGBD.

Examples:
  # Single pair
  python run_any2full.py \
    --rgb /path/to/rgb.png \
    --depth /path/to/depth.png \
    --checkpoint /path/to/ours_checkpoint.pth \
    --out_dir ./outputs

  # Batch (match by basename)
  python run_any2full.py \
    --rgb_dir /path/to/rgb_dir \
    --depth_dir /path/to/depth_dir \
    --checkpoint /path/to/ours_checkpoint.pth \
    --out_dir ./outputs
"""

import argparse
import glob
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
import matplotlib

from model.ours.any2full import Any2Full
from utils.denoise import remove_outliers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Any2Full RGBD inference")

    parser.add_argument("--rgb", type=str, default=None, help="Path to RGB image")
    parser.add_argument("--depth", type=str, default=None, help="Path to depth image or .npy")

    parser.add_argument("--rgb_dir", type=str, default=None, help="Directory of RGB images")
    parser.add_argument("--depth_dir", type=str, default=None, help="Directory of depth images or .npy")

    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--da_ckpt_path", type=str, default=None, help="Optional DA encoder checkpoint")
    parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--init_scailing", type=bool, default=True)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--max_depth", type=float, default=1e3)
    parser.add_argument("--min_depth", type=float, default=1e-6)

    parser.add_argument("--out_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--grayscale", action="store_true", help="Save grayscale depth PNG")

    parser.add_argument("--depth_scale", type=float, default=100.0, help="Scale for depth images (depth = img / scale)")
    parser.add_argument("--denoise", action="store_true", help="Denoise sparse depth before inference")
    parser.add_argument("--denoise_threshold", type=float, default=2.0, help="Outlier threshold (std multiplier)")
    parser.add_argument("--denoise_kernel_size", type=int, default=None, help="Denoise kernel size (odd int)")
    parser.add_argument("--denoise_min_valid", type=int, default=5, help="Min valid neighbors for denoise")

    return parser.parse_args()


def _load_rgb(path: str) -> torch.Tensor:
    rgb_img = Image.open(path).convert("RGB")
    t_rgb = T.Compose([
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return t_rgb(rgb_img).unsqueeze(0)


def _load_depth(path: str, depth_scale: float) -> torch.Tensor:
    if path.lower().endswith(".npy"):
        depth_arr = np.load(path).astype(np.float32)
    else:
        depth_img = Image.open(path)
        depth_arr = np.array(depth_img).astype(np.float32)
        depth_arr = depth_arr / depth_scale

    # Normalize shape to HxW for PIL
    if depth_arr.ndim == 4:
        # (B, C, H, W) -> (H, W)
        depth_arr = depth_arr[0, 0]
    elif depth_arr.ndim == 3:
        # (C, H, W) or (H, W, C) -> (H, W)
        if depth_arr.shape[0] in (1, 3):
            depth_arr = depth_arr[0]
        else:
            depth_arr = depth_arr[:, :, 0]

    depth_img = Image.fromarray(depth_arr.astype(np.float32))
    t_dep = T.ToTensor()
    return t_dep(depth_img).unsqueeze(0)


def _save_depth_outputs(depth: np.ndarray, out_base: Path, grayscale: bool) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_base) + ".npy", depth)

    depth_min = float(np.min(depth))
    depth_max = float(np.max(depth))
    if depth_max > depth_min:
        depth_norm = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = np.zeros_like(depth)

    depth_img = (depth_norm * 255.0).astype(np.uint8)
    if grayscale:
        depth_img = np.repeat(depth_img[..., np.newaxis], 3, axis=-1)
    else:
        cmap = matplotlib.colormaps.get_cmap("Spectral_r")
        depth_img = (cmap(depth_img)[:, :, :3] * 255).astype(np.uint8)

    Image.fromarray(depth_img).save(str(out_base) + ".png")


def _load_checkpoint(model: Any2Full, ckpt_path: str, device: str) -> Any2Full:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    cleaned = OrderedDict((k.replace("module.", ""), v) for k, v in state.items())
    model.load_state_dict(cleaned, strict=True)
    return model


def _collect_pairs(rgb_dir: str, depth_dir: str):
    rgb_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        rgb_files.extend(glob.glob(os.path.join(rgb_dir, ext)))

    depth_files = []
    for ext in ("*.png", "*.npy"):
        depth_files.extend(glob.glob(os.path.join(depth_dir, ext)))

    depth_map = {Path(p).stem: p for p in depth_files}
    pairs = []
    for rgb in sorted(rgb_files):
        key = Path(rgb).stem
        dep = depth_map.get(key)
        if dep is not None:
            pairs.append((rgb, dep))
    return pairs


def main() -> None:
    args = parse_args()

    if (args.rgb and args.depth) and (args.rgb_dir or args.depth_dir):
        raise ValueError("Use either single pair (--rgb/--depth) or batch (--rgb_dir/--depth_dir), not both.")

    if args.rgb and not args.depth:
        raise ValueError("--depth is required when --rgb is provided.")
    if args.depth and not args.rgb:
        raise ValueError("--rgb is required when --depth is provided.")

    if (args.rgb_dir and not args.depth_dir) or (args.depth_dir and not args.rgb_dir):
        raise ValueError("Both --rgb_dir and --depth_dir are required for batch mode.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Any2Full(
        encoder=args.encoder,
        da_ckpt_path=args.da_ckpt_path,
        args=args,
    )
    model = _load_checkpoint(model, args.checkpoint, device)
    model = model.to(device).eval()

    out_dir = Path(args.out_dir)

    if args.rgb and args.depth:
        pairs = [(args.rgb, args.depth)]
    else:
        pairs = _collect_pairs(args.rgb_dir, args.depth_dir)

    if not pairs:
        raise RuntimeError("No RGBD pairs found.")

    for idx, (rgb_path, depth_path) in enumerate(pairs, 1):
        print(f"[{idx}/{len(pairs)}] {rgb_path} | {depth_path}")

        rgb = _load_rgb(rgb_path).to(device)
        dep = _load_depth(depth_path, args.depth_scale)
        if args.denoise:
            dep = remove_outliers(
                dep,
                min_valid=args.denoise_min_valid,
                threshold=args.denoise_threshold,
                kernel_size=args.denoise_kernel_size,
            )
        dep = dep.to(device)

        sample = {"rgb": rgb, "dep": dep}
        with torch.no_grad():
            pred = model(sample)["pred"].squeeze(0).squeeze(0).cpu().numpy()

        out_base = out_dir / Path(rgb_path).stem
        _save_depth_outputs(pred, out_base, args.grayscale)


if __name__ == "__main__":
    main()
