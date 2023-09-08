from typing import Union, Tuple, List
import numpy as np
import cupy as cp

size_2_t = Union[int, Tuple[int, int], List[int]]
array_t = Union[np.ndarray, cp.ndarray]


def get_pair(x: size_2_t):
    if isinstance(x, int):
        return x, x
    assert len(x) == 2
    return tuple(x)

def get_array_module(x: array_t):
    assert isinstance(x, (np.ndarray, cp.ndarray))
    if isinstance(x, np.ndarray):
        return np
    return cp
