<div align="center">
<h1>Any2Full</h1>

Zhiyuan Zhou<sup>1</sup> · Ruofeng Liu<sup>2</sup> · Taichi Liu<sup>1</sup> · Weijian Zuo<sup>3</sup> · Shanshan Wang<sup>1</sup> · Zhiqing Hong<sup>4</sup> · Desheng Zhang<sup>1</sup>
<br>
<sup>1</sup>Rutgers Univ.&emsp;&emsp;&emsp;<sup>2</sup>Michigan State Univ.&emsp;&emsp;&emsp;<sup>3</sup>JD Logistics&emsp;&emsp;&emsp;<sup>4</sup>HKUST (GZ)

<a href="https://arxiv.org/abs/2603.05711"><img src="https://img.shields.io/badge/arXiv-Any2Full-red" alt="Paper PDF"></a>
<a href="https://github.com/zhiyuandaily/Any2Full"><img src="https://img.shields.io/badge/Code-GitHub-green" alt="Code"></a>
<a href="https://huggingface.co/spaces/zhiyuandaily/Any2Full"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" alt="Hugging Face Demo"></a>
<a href="https://huggingface.co/zhiyuandaily/Any2Full/tree/main/checkpoints"><img src="https://img.shields.io/badge/Weights-Download-yellow" alt="Model Weights"></a>
</div>



---

## Overview
![teaser](assets/Any2Full.png)
Accurate dense depth is essential for robotics, but commodity RGBD sensors are often sparse or incomplete. **Any2Full** is a one-stage, domain-general, and pattern-agnostic depth completion framework. It reformulates completion as **scale-prompting adaptation** of a pretrained monocular depth estimation (MDE) model, so the model keeps strong geometric priors while adapting to diverse sparse depth patterns.

## Highlights
- **One-stage scale prompting**: achieves domain-general depth completion by fusing pretrained MDE priors.
- **Scale-Aware Prompt Encoder**: strong robustness under different sparsity levels and sampling patterns.
- **lightweight design**: efficient inference with a single forward pass.

---

## Requirements (Minimal for Inference)
- python==3.9.x
- torch==2.0.1
- torchvision==0.17.2
- numpy
- pillow
- matplotlib
- scipy
- opencv-python

```bash
pip install torch==2.0.1 torchvision==0.17.2 \
  numpy pillow matplotlib scipy opencv-python
```

---

## Model Usage

### 1) Quick Inference (Single or Batch RGBD)
Use `run_any2full.py` for single RGBD pairs or batch folders (matched by filename stem).

Example inputs are provided under `assets/`: `assets/rgb` and `assets/depth` can be used as inputs, and `assets/output` shows the corresponding outputs.

```bash
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
```

Optional denoise (from `utils/denoise.py`): Any2Full relies on accurate sparse depth as an anchor, so cleaner raw depth generally yields better results. We provide a simple denoising pre-processing step for convenience.
```bash
python run_any2full.py \
  --rgb /path/to/rgb.png \
  --depth /path/to/depth.png \
  --checkpoint /path/to/ours_checkpoint.pth \
  --out_dir ./outputs \
  --denoise \
  --denoise_threshold 2 \
  --denoise_kernel_size 9
```

---
### Inference Parameters (Detailed)
- `--rgb`: RGB image path (single mode).
- `--depth`: Sparse depth path (`.png` or `.npy`) (single mode).
- `--rgb_dir`: RGB directory (batch mode, filename stem matched).
- `--depth_dir`: Depth directory (batch mode, filename stem matched).
- `--checkpoint`: Any2Full checkpoint path (required).
- `--da_ckpt_path`: Optional backbone MDE checkpoint (for encoder init).
- `--encoder`: Backbone variant (`vits`, `vitb`, `vitl`).
- `--depth_scale`: Scale factor for depth PNGs (depth = img / scale).
- `--denoise`: Enable sparse depth outlier removal before inference.
- `--denoise_threshold`: Outlier threshold (std multiplier).
- `--denoise_kernel_size`: Neighborhood size (odd int) for denoise; `None` auto-estimates.
- `--denoise_min_valid`: Minimum valid neighbors for denoise.

### Model Weights
- **Any2Full weights**: https://huggingface.co/zhiyuandaily/Any2Full/tree/main/checkpoints

---

## Citation
If you find our work useful, please consider citing:
```
@article{zhou2026any2full,
  title={Any to Full: Prompting Depth Anything for Depth Completion in One Stage},
  author={Zhou, Zhiyuan and Liu, Ruofeng and Liu, Taichi and Zuo, Weijian and Wang, Shanshan and Hong, Zhiqing and Zhang, Desheng},
  journal={arXiv:2603.05711},
  year={2026}
}
```
