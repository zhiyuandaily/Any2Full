import numpy as np
from scipy.ndimage import generic_filter


def _to_numpy_2d(sparse_depth):
    """
    Convert input to numpy float32.
    Returns: (arr, input_type, original_shape, torch_dtype)
    """
    try:
        import torch
        if isinstance(sparse_depth, torch.Tensor):
            arr = sparse_depth.detach().cpu().numpy().astype(np.float32)
            return arr, 'torch', sparse_depth.shape, sparse_depth.dtype
    except ImportError:
        pass

    arr = np.asarray(sparse_depth, dtype=np.float32)
    return arr, 'numpy', sparse_depth.shape, None


def _to_3d(arr):
    """Reshape to (N, H, W). Returns (arr_3d, original_ndim, original_shape)."""
    original_ndim = arr.ndim
    original_shape = arr.shape

    if arr.ndim == 2:
        return arr[np.newaxis], original_ndim, original_shape
    elif arr.ndim == 3:
        return arr, original_ndim, original_shape
    elif arr.ndim == 4:
        B, C, H, W = arr.shape
        return arr.reshape(B * C, H, W), original_ndim, original_shape
    else:
        raise ValueError(f"Unsupported shape: {arr.shape}")


def _restore_shape(result, original_ndim, original_shape):
    """Restore original shape."""
    if original_ndim == 2:
        return result[0]
    else:
        return result.reshape(original_shape)


def _get_kernel_size(arr, min_valid):
    """
    Estimate minimal kernel size to include at least min_valid neighbors.

    Core formula:
        area_needed = min_valid / valid_ratio
        k = ceil(sqrt(area_needed))
    """
    valid_ratio = np.sum(arr > 0) / arr.size
    if valid_ratio <= 0:
        return None  # empty frame

    area_needed = min_valid / valid_ratio
    k = int(np.ceil(np.sqrt(area_needed)))
    k = k if k % 2 == 1 else k + 1  # ensure odd
    k = max(3, min(k, 41))           # clamp to [3, 41]
    return k


def _remove_outliers_2d(arr, min_valid, threshold, kernel_size):
    """
    threshold: outlier if deviation > mean_dev + threshold * std_dev
    """
    def is_outlier(values):
        center = values[len(values) // 2]
        if center == 0:
            return center                        # invalid point

        valid = values[values > 0]
        if len(valid) < min_valid:
            return 0                             # isolated point

        neighbor_median = np.median(valid)
        devs = np.abs(valid - neighbor_median)   # neighbor deviations

        mean_dev = np.mean(devs)
        std_dev  = np.std(devs)

        center_dev = abs(center - neighbor_median)
        if center_dev > mean_dev + threshold * std_dev:
            return 0                             # outlier
        return center                            # keep

    return generic_filter(arr.astype(np.float32), is_outlier, size=kernel_size)


def remove_outliers(sparse_depth, min_valid=5, threshold=2,
                    kernel_size=None, verbose=False):
    """
    Remove outliers in sparse depth (isolated points & abnormal depths) without filling.

    Args:
        sparse_depth : numpy array or torch.Tensor
                       shapes: (H, W) / (1, H, W) / (B, H, W) / (B, 1, H, W)
        min_valid    : minimum valid neighbors; below this treated as isolated
        threshold    : deviation threshold; above is treated as outlier
        kernel_size  : neighborhood size (odd). None auto-estimates per frame
        verbose      : log sparsity and kernel size per frame

    Returns:
        Denoised result with the same type and shape as input
    """
    arr, input_type, original_shape, torch_dtype = _to_numpy_2d(sparse_depth)
    arr_3d, original_ndim, _ = _to_3d(arr)

    processed = []
    for i, frame in enumerate(arr_3d):
        # Determine kernel size
        k = kernel_size if kernel_size is not None else _get_kernel_size(frame, min_valid)

        if k is None:
            # Skip empty frames
            processed.append(frame)
            continue

        if verbose:
            valid_ratio = np.mean(frame > 0)
            print(f"[Denoise] frame={i}, valid_ratio={valid_ratio:.3%}, kernel={k}x{k}")

        processed.append(_remove_outliers_2d(frame, min_valid, threshold, k))

    result = np.stack(processed, axis=0)
    result = _restore_shape(result, original_ndim, original_shape)

    # Restore original type
    if input_type == 'torch':
        import torch
        result = torch.from_numpy(result).to(dtype=torch_dtype,
                                             device=sparse_depth.device)
    return result
