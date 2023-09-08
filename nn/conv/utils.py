from typing import Union, Tuple, List

stride_type = Union[int, Tuple[int, int], List[int]]


def get_stride(stride: stride_type) -> Tuple[int, int]:
    if isinstance(stride, int):
        return stride, stride
    assert len(stride) == 2
    return tuple(stride)
