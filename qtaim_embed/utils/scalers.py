import logging
from typing import Optional
import numpy as np
from sklearn.preprocessing import StandardScaler as sk_StandardScaler
import torch

logger = logging.getLogger(__name__)



def guard_std(std, mean=None, rel_tol: float = 10.0):
    """Set the scale of constant and float-noise-constant columns to 1.0.

    A column is constant when std <= rel_tol * eps64 * max(|mean|, 1), the same
    relative test sklearn uses in _is_constant_feature, so fit, apply, merge and
    inverse all agree and a column that is constant up to rounding never divides
    by ~1e-17. Genuine small variances (std 1e-3 and up) are untouched. Accepts
    torch tensors or numpy arrays and returns the same type, modified in place.
    """
    eps = torch.finfo(torch.float64).eps
    if isinstance(std, np.ndarray):
        m = np.ones_like(std) if mean is None else np.maximum(np.abs(np.asarray(mean)), 1.0)
        std[std <= rel_tol * eps * m] = 1.0
        return std
    m = torch.ones_like(std) if mean is None else torch.clamp(mean.abs().to(std.dtype), min=1.0)
    std[std <= rel_tol * eps * m] = 1.0
    return std

def compute_running_average(
    old_avg: float, new_value: float, n: int, n_new: Optional[int] = 1
) -> float:
    """simple running average
    Args:
        old_avg (float): old average
        new_value (float): new value
        n (int): number of samples
        n_new (Optional[int]): number of new samples
    """
    if n == 0:
        return new_value
    if n_new == 0:
        return old_avg

    return old_avg + (new_value - old_avg) * n_new / (n + n_new)


def _transform(
    X: torch.Tensor,
    copy: bool,
    with_mean: bool = True,
    with_std: bool = True,
    threshold: float = 1.0e-3,
    eta: float = 1.0e-3,
):
    """
    Args:
        X: a list of 1D tensor or a 2D tensor
    Returns:
        rst: 2D array
        mean: 1D array
        std: 1D array
    """
    if isinstance(X, list):
        X = torch.stack(X)
    scaler = sk_StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)
    rst = scaler.fit(X)
    mean = scaler.mean_
    std = np.sqrt(scaler.var_)
    # print("mean", mean)
    # print("std", std)
    for i, v in enumerate(std):
        if v <= threshold:
            logger.warning(
                "Standard deviation for feature %d is %s, smaller than %s. "
                "You may want to exclude this feature.", i, v, threshold
            )

    rst = scaler.transform(X)
    # sklearn's transform divides constant columns by 1.0 internally
    # (_handle_zeros_in_scale, relative tolerance); the returned std must match
    # what was actually used or later apply/inverse calls disagree
    std = guard_std(std, mean)

    return rst, mean, std
