from typing import Optional
import numpy as np 
from sklearn.preprocessing import StandardScaler as sk_StandardScaler
import torch 


def compute_running_average(
    old_avg: float, 
    new_value: float, 
    n: int, 
    n_new: Optional[int] = 1
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
        with_mean: bool=True, 
        with_std: bool=True, 
        threshold: float=1.0e-3, 
        eta: float=1.0e-3
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
            print(
                "Standard deviation for feature {} is {}, smaller than {}. "
                "You may want to exclude this feature.".format(i, v, threshold)
            )

    rst = scaler.transform(X)
    # make all values < eta in std to be eta
    std[std < eta] = eta
    # manually scale the data
    # rst = (rst - mean) / std

    return rst, mean, std





