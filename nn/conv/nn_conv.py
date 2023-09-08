import numpy as np
from nn.conv.utils import stride_type, get_stride


def conv2d_nhwc_v1(img: np.ndarray, kernel: np.ndarray, stride: stride_type) -> np.ndarray:
    batch_size, ih, iw, _ic = img.shape
    kh, kw, ic, oc = kernel.shape
    assert _ic == ic
    sh, sw = get_stride(stride)
    oh = (ih - kh) // sh + 1
    ow = (iw - kw) // sw + 1
    istepb, isteph, istepw, istepc = img.strides
    x = np.lib.stride_tricks.as_strided(img, (batch_size, oh, ow, kh, kw, ic),
                                        (istepb, isteph * sh, istepw * sw, isteph, istepw, istepc), writeable=False)
    xw = np.tensordot(x, kernel, ([-3, -2, -1], [0, 1, 2]))
    return xw


def conv2d_nchw_v1(img: np.ndarray, kernel: np.ndarray, stride: stride_type) -> np.ndarray:
    batch_size, _ic, ih, iw = img.shape
    oc, ic, kh, kw = kernel.shape
    assert _ic == ic
    sh, sw = get_stride(stride)
    oh = (ih - kh) // sh + 1
    ow = (iw - kw) // sw + 1
    istepb, istepc, isteph, istepw = img.strides
    x = np.lib.stride_tricks.as_strided(img, (batch_size, ic, oh, ow, kh, kw),
                                        (istepb, istepc, isteph * sh, istepw * sw, isteph, istepw), writeable=False)
    xw = np.tensordot(x, kernel, ([1, 4, 5], [1, 2, 3]))
    return np.transpose(xw, (0, 3, 1, 2))


def conv2d_nchw_v2(img: np.ndarray, kernel: np.ndarray, stride: stride_type) -> np.ndarray:
    batch_size, _ic, ih, iw = img.shape
    oc, ic, kh, kw = kernel.shape
    assert _ic == ic
    sh, sw = get_stride(stride)
    oh = (ih - kh) // sh + 1
    ow = (iw - kw) // sw + 1
    l0 = np.arange(oh)[:, None, None, None]
    l1 = np.arange(ow)[None, :, None, None]
    l2 = np.arange(kh)[None, None, :, None]
    l3 = np.arange(kw)[None, None, None, :]
    # batch_size, ic, oh, ow, kh, kw
    x = img[:, :, l0 * sh + l2, l1 * sw + l3]
    xw = np.tensordot(x, kernel, ([1, 4, 5], [1, 2, 3]))
    return np.transpose(xw, (0, 3, 1, 2))


def _check():
    import time
    import torch
    import torch.nn.functional as F
    def conv2d_torch(img: np.ndarray, kernel: np.ndarray, stride: stride_type) -> np.ndarray:
        x = torch.from_numpy(img)
        w = torch.from_numpy(kernel)
        return F.conv2d(x, w, bias=None, stride=stride).numpy()

    x = np.random.rand(16, 128, 200, 210).astype(np.float32)
    k = np.random.rand(512, 128, 3, 3).astype(np.float32)

    x_cl = np.transpose(x, (0, 2, 3, 1))
    k_cl = np.transpose(k, (2, 3, 1, 0))

    s = (3, 5)
    tic = time.time()
    rt = conv2d_torch(x, k, s)
    print(f'{(time.time() - tic) * 1000:.4f} ms')
    print(rt.shape)

    tic = time.time()
    r1 = conv2d_nchw_v1(x, k, s)
    print(f'{(time.time() - tic) * 1000:.4f} ms')
    print(r1.shape, np.abs(r1 - rt).max())

    tic = time.time()
    r2 = conv2d_nchw_v2(x, k, s)
    print(f'{(time.time() - tic) * 1000:.4f} ms')
    print(r2.shape, np.abs(r2 - rt).max())

    tic = time.time()
    r1_cl = conv2d_nhwc_v1(x_cl, k_cl, s)
    print(f'{(time.time() - tic) * 1000:.4f} ms')
    r1_cl2cf = np.transpose(r1_cl, (0, 3, 1, 2))
    print(r1_cl2cf.shape, np.abs(r1_cl2cf - rt).max())


if __name__ == '__main__':
    _check()
