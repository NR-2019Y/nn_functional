import numpy as np
from nn.conv.utils import stride_type, get_stride


def conv_nchw_transpose2d(img: np.ndarray, kernel: np.ndarray, stride: stride_type) -> np.ndarray:
    batch_size, _ic, ih, iw = img.shape
    ic, oc, kh, kw = kernel.shape
    assert _ic == ic
    sh, sw = get_stride(stride)
    oh = (ih - 1) * sh + kh
    ow = (iw - 1) * sw + kw
    imcol = np.zeros((batch_size, ic, oh, ow, kh, kw), dtype=img.dtype)
    l0 = np.arange(0, ih * sh, sh)[:, None, None, None]
    l1 = np.arange(0, iw * sw, sw)[None, :, None, None]
    l2 = np.arange(kh)[None, None, :, None]
    l3 = np.arange(kw)[None, None, None, :]
    imcol[:, :, l0 + l2, l1 + l3, l2, l3] = img[:, :, :, :, None, None]
    xw = np.tensordot(imcol, kernel, ([1, 4, 5], [0, 2, 3]))
    return np.transpose(xw, (0, 3, 1, 2))


def _check():
    import time
    import torch
    import torch.nn.functional as F
    def convt_torch(img: np.ndarray, kernel: np.ndarray, stride: stride_type) -> np.ndarray:
        x = torch.from_numpy(img)
        w = torch.from_numpy(kernel)
        r = F.conv_transpose2d(x, w, bias=None, stride=stride)
        return r.numpy()

    x = np.random.rand(2, 64, 100, 100).astype(np.float32)
    w = np.random.rand(64, 256, 3, 3).astype(np.float32)
    s = (4, 4)

    tic = time.time()
    rt = convt_torch(x, w, s)
    print(f'{(time.time() - tic) * 1000:.4f} ms')
    print(rt.shape)

    tic = time.time()
    r1 = conv_nchw_transpose2d(x, w, s)
    print(f'{(time.time() - tic) * 1000:.4f} ms')
    print(r1.shape, np.abs(r1 - rt).max())


if __name__ == '__main__':
    _check()
