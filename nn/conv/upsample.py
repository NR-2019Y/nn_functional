from nn.conv.utils import array_t, get_array_module


def upsample_bilinear(x: array_t, scale_factor: float = 2.0):
    xp = get_array_module(x)
    _, _, h, w = x.shape
    rh = xp.linspace(0, h - 1, int(h * scale_factor), dtype=xp.float32)[:, None]
    rw = xp.linspace(0, w - 1, int(w * scale_factor), dtype=xp.float32)[None, :]
    y1, y2 = xp.floor(rh), xp.ceil(rh)
    x1, x2 = xp.floor(rw), xp.ceil(rw)
    a = rw - x1
    b = rh - y1
    neg_a = 1 - a
    neg_b = 1 - b
    y1i = y1.astype(xp.int32)
    y2i = y2.astype(xp.int32)
    x1i = x1.astype(xp.int32)
    x2i = x2.astype(xp.int32)
    fa = x[..., y1i, x1i]
    fb = x[..., y1i, x2i]
    fc = x[..., y2i, x1i]
    fd = x[..., y2i, x2i]
    return neg_a * neg_b * fa + a * neg_b * fb + neg_a * b * fc + a * b * fd


def _check():
    # import numpy as np
    import cupy as np
    import torch
    import time
    from torch import nn
    import cupy
    from cupy._core.dlpack import toDlpack
    from torch.utils.dlpack import from_dlpack, to_dlpack
    def up_torch(x: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        up_func = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        x = from_dlpack(toDlpack(x))
        return cupy.from_dlpack(to_dlpack(up_func(x)))
        # x_tensor = torch.from_numpy(x)
        # return up_func(x_tensor).numpy()

    x = np.random.rand(1, 128, 320, 320).astype(np.float32)
    for _ in range(20):
        tic = time.time()
        r1 = upsample_bilinear(x, 2.0)
        print(f'numpy-{(time.time() - tic) * 1000:.4f} ms')

    for _ in range(20):
        tic = time.time()
        rt = up_torch(x, 2.0)
        print(f'torch-{(time.time() - tic) * 1000:.4f} ms')
    print(np.abs(r1 - rt).max())


if __name__ == '__main__':
    _check()
