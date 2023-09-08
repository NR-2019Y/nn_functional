# import numpy as np
import cupy
import cupy as cp
from nn.conv.utils import stride_type, get_stride


def conv2d_nhwc_v1(img: cp.ndarray, kernel: cp.ndarray, stride: stride_type) -> cp.ndarray:
    batch_size, ih, iw, _ic = img.shape
    kh, kw, ic, oc = kernel.shape
    assert _ic == ic
    sh, sw = get_stride(stride)
    oh = (ih - kh) // sh + 1
    ow = (iw - kw) // sw + 1
    istepb, isteph, istepw, istepc = img.strides
    x = cp.lib.stride_tricks.as_strided(img, (batch_size, oh, ow, kh, kw, ic),
                                        (istepb, isteph * sh, istepw * sw, isteph, istepw, istepc))
    xw = cp.tensordot(x, kernel, ([-3, -2, -1], [0, 1, 2]))
    return xw


def conv2d_nchw_v1(img: cp.ndarray, kernel: cp.ndarray, stride: stride_type) -> cp.ndarray:
    batch_size, _ic, ih, iw = img.shape
    oc, ic, kh, kw = kernel.shape
    assert _ic == ic
    sh, sw = get_stride(stride)
    oh = (ih - kh) // sh + 1
    ow = (iw - kw) // sw + 1
    istepb, istepc, isteph, istepw = img.strides
    x = cp.lib.stride_tricks.as_strided(img, (batch_size, ic, oh, ow, kh, kw),
                                        (istepb, istepc, isteph * sh, istepw * sw, isteph, istepw))
    # x = cp.ascontiguousarray(x)
    xw = cp.tensordot(x, kernel, ([1, 4, 5], [1, 2, 3]))
    return cp.transpose(xw, (0, 3, 1, 2))


def conv2d_nchw_v2(img: cp.ndarray, kernel: cp.ndarray, stride: stride_type) -> cp.ndarray:
    batch_size, _ic, ih, iw = img.shape
    oc, ic, kh, kw = kernel.shape
    assert _ic == ic
    sh, sw = get_stride(stride)
    oh = (ih - kh) // sh + 1
    ow = (iw - kw) // sw + 1
    l0 = cp.arange(oh)[:, None, None, None]
    l1 = cp.arange(ow)[None, :, None, None]
    l2 = cp.arange(kh)[None, None, :, None]
    l3 = cp.arange(kw)[None, None, None, :]
    # batch_size, ic, oh, ow, kh, kw
    x = img[:, :, l0 * sh + l2, l1 * sw + l3]
    # x = cp.ascontiguousarray(x)
    xw = cp.tensordot(x, kernel, ([1, 4, 5], [1, 2, 3]))
    return cp.transpose(xw, (0, 3, 1, 2))


def _check():
    import time
    import torch
    import torch.nn.functional as F
    from cupy._core.dlpack import toDlpack
    from torch.utils.dlpack import to_dlpack, from_dlpack
    def conv2d_torch(img: cp.ndarray, kernel: cp.ndarray, stride: stride_type) -> cp.ndarray:
        x = from_dlpack(toDlpack(img))
        w = from_dlpack(toDlpack(kernel))
        # tic = time.time()
        r = F.conv2d(x, w, bias=None, stride=stride)
        # print(f'F.conv2d {(time.time() - tic) * 1000:.4f} ms')
        return cupy.from_dlpack(to_dlpack(r))

    x = cp.random.rand(16, 128, 200, 210).astype(cp.float32)
    k = cp.random.rand(512, 128, 3, 3).astype(cp.float32)

    x_cl = cp.transpose(x, (0, 2, 3, 1))
    k_cl = cp.transpose(k, (2, 3, 1, 0))
    s = (3, 5)

    WUP = 0

    [conv2d_torch(x[:0], k, s) for _ in range(WUP)]
    for i in range(4):
        tic = time.time()
        rt = conv2d_torch(x, k, s)
        print(f'{i}-{(time.time() - tic) * 1000:.4f} ms')
        # print(rt.shape)

    [conv2d_nchw_v1(x[:0], k, s) for _ in range(WUP)]
    for i in range(4):
        tic = time.time()
        r1 = conv2d_nchw_v1(x, k, s)
        print(f'{i}-{(time.time() - tic) * 1000:.4f} ms')
        # print(r1.shape, cp.abs(r1 - rt).max())

    [conv2d_nchw_v2(x[:0], k, s) for _ in range(WUP)]
    for i in range(4):
        tic = time.time()
        r2 = conv2d_nchw_v2(x, k, s)
        print(f'{i}-{(time.time() - tic) * 1000:.4f} ms')
        # print(r2.shape, cp.abs(r2 - rt).max())

    [conv2d_nhwc_v1(x_cl[:0], k_cl, s) for _ in range(WUP)]
    for i in range(4):
        tic = time.time()
        r1_cl = conv2d_nhwc_v1(x_cl, k_cl, s)
        print(f'{i}-{(time.time() - tic) * 1000:.4f} ms')
        r1_cl2cf = cp.transpose(r1_cl, (0, 3, 1, 2))
        # print(r1_cl2cf.shape, cp.abs(r1_cl2cf - rt).max())


if __name__ == '__main__':
    _check()
