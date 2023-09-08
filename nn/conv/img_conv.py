import itertools
import numpy as np
from nn.conv.utils import stride_type, get_stride

# reference: github.com/sebgao/cTensor

def img_conv_v1(img: np.ndarray, kernel: np.ndarray, stride: stride_type) -> np.ndarray:
    assert img.ndim == 2
    assert kernel.ndim == 2
    ih, iw = img.shape
    kh, kw = kernel.shape
    sh, sw = get_stride(stride)
    oh = (ih - kh) // sh + 1
    ow = (iw - kw) // sw + 1
    isteph, istepw = img.strides
    x = np.lib.stride_tricks.as_strided(img, (oh, ow, kh, kw), (isteph * sh, istepw * sw, isteph, istepw),
                                        writeable=False)
    return np.tensordot(x, kernel, ([-2, -1], [0, 1]))


def img_conv_v2(img: np.ndarray, kernel: np.ndarray, stride: stride_type) -> np.ndarray:
    assert img.ndim == 2
    assert kernel.ndim == 2
    ih, iw = img.shape
    kh, kw = kernel.shape
    sh, sw = get_stride(stride)
    oh = (ih - kh) // sh + 1
    ow = (iw - kw) // sw + 1
    l0 = np.arange(oh)[:, None, None, None]
    l1 = np.arange(ow)[None, :, None, None]
    l2 = np.arange(kh)[None, None, :, None]
    l3 = np.arange(kw)[None, None, None, :]
    x = img[l0 * sh + l2, l1 * sw + l3]
    return np.tensordot(x, kernel, ([-2, -1], [0, 1]))


def img_conv_transpose_v1(img: np.ndarray, kernel: np.ndarray, stride: stride_type) -> np.ndarray:
    assert img.ndim == 2
    assert kernel.ndim == 2
    ih, iw = img.shape
    kh, kw = kernel.shape
    sh, sw = get_stride(stride)
    oh = (ih - 1) * sh + kh
    ow = (iw - 1) * sw + kw
    imcol = np.zeros((oh, ow, kh, kw), dtype=img.dtype)
    l0 = np.arange(0, ih * sh, sh)[:, None, None, None]
    l1 = np.arange(0, iw * sw, sw)[None, :, None, None]
    l2 = np.arange(kh)[None, None, :, None]
    l3 = np.arange(kw)[None, None, None, :]
    imcol[l0 + l2, l1 + l3, l2, l3] = img[..., None, None]
    return np.tensordot(imcol, kernel, ([-2, -1], [0, 1]))


def img_conv_transpose_v2(img: np.ndarray, kernel: np.ndarray, stride: stride_type) -> np.ndarray:
    assert img.ndim == 2
    assert kernel.ndim == 2
    ih, iw = img.shape
    kh, kw = kernel.shape
    sh, sw = get_stride(stride)
    oh = (ih - 1) * sh + kh
    ow = (iw - 1) * sw + kw
    imcol = np.zeros((oh, ow, kh, kw), dtype=img.dtype)
    for i, j in itertools.product(range(kh), range(kw)):
        imcol[i:i + sh * ih:sh, j:j + sw * iw:sw, i, j] = img
    return np.tensordot(imcol, kernel, ([-2, -1], [0, 1]))


def _check_calc():
    import torch
    import time
    import torch.nn.functional as F
    def img_conv_torch(img: np.ndarray, kernel: np.ndarray, stride: stride_type) -> np.ndarray:
        x = torch.from_numpy(img[None, None])
        w = torch.from_numpy(kernel[None, None])
        r = F.conv2d(x, w, bias=None, stride=stride)
        return r[0, 0].numpy()

    x = np.random.rand(400, 411)
    w = np.random.rand(271, 200)
    s = (2, 3)
    tic = time.time()
    rt = img_conv_torch(x, w, s)
    print(f'{(time.time() - tic) * 1000:.4f} ms')
    print(rt.shape)

    tic = time.time()
    r1 = img_conv_v1(x, w, s)
    print(f'{(time.time() - tic) * 1000:.4f} ms')
    print(r1.shape, np.abs(r1 - rt).max())

    tic = time.time()
    r2 = img_conv_v2(x, w, s)
    print(f'{(time.time() - tic) * 1000:.4f} ms')
    print(r2.shape, np.abs(r2 - rt).max())


def _check_calc_transpose():
    import time
    import torch
    import torch.nn.functional as F
    def convt_torch(img: np.ndarray, kernel: np.ndarray, stride: stride_type) -> np.ndarray:
        x = torch.from_numpy(img[None, None])
        w = torch.from_numpy(kernel[None, None])
        r = F.conv_transpose2d(x, w, bias=None, stride=stride)
        return r[0, 0].numpy()

    x = np.random.rand(400, 411)
    w = np.random.rand(27, 47)
    s = (2, 3)
    tic = time.time()
    rt = convt_torch(x, w, s)
    print(f'{(time.time() - tic) * 1000:.4f} ms')
    print(rt.shape)

    tic = time.time()
    r1 = img_conv_transpose_v1(x, w, s)
    print(f'{(time.time() - tic) * 1000:.4f} ms')
    print(r1.shape, np.abs(r1 - rt).max())

    tic = time.time()
    r2 = img_conv_transpose_v2(x, w, s)
    print(f'{(time.time() - tic) * 1000:.4f} ms')
    print(r2.shape, np.abs(r2 - rt).max())


if __name__ == '__main__':
    # _check_calc()
    _check_calc_transpose()
