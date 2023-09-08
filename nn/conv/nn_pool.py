from nn.conv.utils import array_t, size_2_t, get_array_module, get_pair


def maxpool2d_nchw(img: array_t, ksize: size_2_t, stride: size_2_t = 1, dilation: size_2_t = 1) -> array_t:
    xp = get_array_module(img)
    batch_size, ic, ih, iw = img.shape
    kh, kw = ksize
    sh, sw = get_pair(stride)
    dh, dw = get_pair(dilation)
    dkh = kh + (kh - 1) * (dh - 1)
    dkw = kw + (kw - 1) * (dw - 1)
    oh = (ih - dkh) // sh + 1
    ow = (iw - dkw) // sw + 1
    istepb, istepc, isteph, istepw = img.strides
    x = xp.lib.stride_tricks.as_strided(img, (batch_size, ic, oh, ow, kh, kw),
                                        (istepb, istepc, isteph * sh, istepw * sw, isteph * dh, istepw * dw))
    x = x.reshape(batch_size, ic, oh, ow, kh * kw)
    return x.max(axis=-1)


def _check_np():
    import time
    import torch
    import torch.nn.functional as F
    import numpy as np

    x = np.random.rand(2, 32, 640, 640)
    x_tensor = torch.from_numpy(x)
    ksize = (3, 5)
    s = (2, 3)
    d = (4, 3)

    tic = time.time()
    rt = F.max_pool2d(x_tensor, ksize, s, 0, d).numpy()
    print(f'torch-{(time.time() - tic) * 1000:.4f} ms')
    print(rt.shape)

    tic = time.time()
    r1 = maxpool2d_nchw(x, ksize, s, d)
    print(f'torch-{(time.time() - tic) * 1000:.4f} ms')
    print(r1.shape, np.abs(rt - r1).max())


def _check_cp():
    import time
    import torch
    import torch.nn.functional as F
    import cupy as cp
    from cupy._core.dlpack import toDlpack
    from cupy import from_dlpack as fromDlpack
    from torch.utils.dlpack import to_dlpack, from_dlpack

    x = cp.random.rand(2, 32, 640, 640)
    x_tensor = from_dlpack(toDlpack(x))
    ksize = (3, 5)
    s = (2, 3)
    d = (4, 3)

    for _ in range(4):
        tic = time.time()
        rt = F.max_pool2d(x_tensor, ksize, s, 0, d)
        print(f'torch-{(time.time() - tic) * 1000:.4f} ms')
    rt = fromDlpack(to_dlpack(rt))
    print(rt.shape)

    for _ in range(4):
        tic = time.time()
        r1 = maxpool2d_nchw(x, ksize, s, d)
        print(f'cupy-{(time.time() - tic) * 1000:.4f} ms')
    print(r1.shape, cp.abs(rt - r1).max())


if __name__ == '__main__':
    _check_np()
    # _check_cp()
