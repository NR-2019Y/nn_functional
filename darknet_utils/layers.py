from nn.conv.utils import size_2_t, array_t, get_pair
import cupy as xp
from nn.conv import act
from nn.conv.nn_conv_cu import conv2d_nchw_v1
from nn.conv.nn_pool import maxpool2d_nchw
from nn.conv.upsample import upsample_bilinear
from collections import OrderedDict
from typing import Sequence, Optional, Dict, Callable
import functools
import operator


def shape2count(shape: Sequence[int]) -> int:
    return functools.reduce(operator.mul, shape)


class DarknetLayer:
    def __init__(self):
        self._modules = OrderedDict()
        self._weights = OrderedDict()

    def add_module(self, name, module):
        self._modules[name] = module

    def add_weights(self, name, weights: array_t):
        self._weights[name] = weights

    def named_weights(self):
        for name, weights in self._weights.items():
            yield name, weights
        for module in self._modules.values():
            yield from module.named_weights()

    def __getattr__(self, item):
        if item in self.__dict__['_weights'].keys():
            return self._weights[item]
        if item in self.__dict__['_modules'].keys():
            return self._modules[item]
        return self.__dict__[item]

    def __call__(self, x: xp.ndarray):
        raise NotImplemented


class LeakyRelu(DarknetLayer):
    def __init__(self, negativa_slope: float = 0.1):
        super().__init__()
        self.negativa_slope = negativa_slope

    def __call__(self, x: xp.ndarray):
        return act.leaky_relu(x, negative_slope=self.negativa_slope)


ACTIVATION_FUNCTION: Dict[str, Optional[Callable]] = {
    'linear': None,
    'leaky': LeakyRelu(negativa_slope=0.1),
    'mish': act.mish
}


class Convolutional(DarknetLayer):
    def __init__(self, in_ch: int, out_ch: int,
                 ksize: size_2_t, stride: size_2_t,
                 act: Optional[DarknetLayer] = None,
                 pad: bool = False, use_bn: bool = True, eps: float = 1e-5):
        super().__init__()
        kh, kw = get_pair(ksize)
        self.ph = (kh - 1) // 2 if pad else 0
        self.pw = (kw - 1) // 2 if pad else 0
        self.stride = get_pair(stride)
        self.act = act
        self.use_bn = use_bn
        self.eps = eps
        if use_bn:
            self.shapes = [
                (out_ch,), (out_ch,), (out_ch,), (out_ch,), (out_ch, in_ch, kh, kw)
            ]
        else:
            self.shapes = [(out_ch,), (out_ch, in_ch, kh, kw)]
        # print("shapes", self.shapes)

    def __call__(self, x: xp.ndarray):
        # print(self.weight.shape, self.bias.shape)
        if self.ph or self.pw:
            x = xp.pad(x, [[0, 0], [0, 0], [self.ph, self.ph], [self.pw, self.pw]])
        x = conv2d_nchw_v1(x, self.weight, self.stride) + self.bias[None, :, None, None]
        if self.act is not None:
            return self.act(x)
        return x

    def _set_weights_bn(self,
                        bn_bias: xp.ndarray,
                        bn_weights: xp.ndarray,
                        bn_running_mean: xp.ndarray,
                        bn_running_var: xp.ndarray,
                        conv_weight: xp.ndarray):
        assert self.use_bn
        bn_std = xp.sqrt(bn_running_var + self.eps)
        inv_bn_std = 1. / bn_std
        gamma_mul_inv_bn_std = bn_weights * inv_bn_std
        new_bias = bn_bias - bn_running_mean * gamma_mul_inv_bn_std
        new_weights = conv_weight * gamma_mul_inv_bn_std[:, None, None, None]
        self.add_weights('weight', new_weights)
        self.add_weights('bias', new_bias)

    def _set_weights_nobn(self, conv_bias: xp.ndarray, conv_weight: xp.ndarray):
        assert not self.use_bn
        self.add_weights('weight', conv_weight)
        self.add_weights('bias', conv_bias)

    @staticmethod
    def get_weights_data_from_bytes(fp, shapes):
        weights = []
        for shape in shapes:
            cnt = shape2count(shape)
            weights.append(xp.fromfile(fp, xp.float32, count=cnt).reshape(shape))
        return weights

    def set_weights_from_bytes(self, fp):
        weights_data = self.get_weights_data_from_bytes(fp, self.shapes)
        if self.use_bn:
            self._set_weights_bn(*weights_data)
        else:
            self._set_weights_nobn(*weights_data)


class MaxPool2D(DarknetLayer):
    def __init__(self, ksize: int, stride: int):
        super(MaxPool2D, self).__init__()
        self.ksize = ksize
        self.stride = stride

    def __call__(self, x: xp.ndarray) -> xp.ndarray:
        # https://github.com/eriklindernoren/PyTorch-YOLOv3.git
        if self.ksize == 2 and self.stride == 1:
            x = xp.pad(x, [[0, 0], [0, 0], [0, 1], [0, 1]])
        else:
            p = (self.ksize - 1) // 2
            if p != 0:
                x = xp.pad(x, [[0, 0], [0, 0], [p, p], [p, p]])
        x = maxpool2d_nchw(x, self.ksize, self.stride)
        return x


class UpsampleBilinear(DarknetLayer):
    def __init__(self, scale_factor: float):
        super(UpsampleBilinear, self).__init__()
        self.scale_factor = scale_factor

    def __call__(self, x: xp.ndarray) -> xp.ndarray:
        return upsample_bilinear(x, scale_factor=self.scale_factor)


class Route(DarknetLayer):
    def __call__(self, x_list: Sequence[array_t]) -> array_t:
        assert isinstance(x_list, (list, tuple))
        if len(x_list) == 1:
            return x_list[0]
        # print([e.shape for e in x_list])
        xout = xp.concatenate(x_list, axis=1)
        # print("route", xout.shape)
        return xout


class ShortCut(DarknetLayer):
    def __init__(self, act: Optional[DarknetLayer]):
        super(ShortCut, self).__init__()
        self.act = act

    def __call__(self, x_list: Sequence[array_t]) -> array_t:
        assert isinstance(x_list, (list, tuple))
        x_data_iter = iter(x_list)
        x = next(x_data_iter)
        for xi in x_data_iter:
            x = x + xi
        if self.act is None:
            return x
        return self.act(x)


# https://github.com/WongKinYiu/PyTorch_YOLOv4/blob/master/utils/layers.py
class Reorg(DarknetLayer):
    def __call__(self, x: array_t):
        return xp.concatenate((
            x[..., 0::2, 0::2],
            x[..., 1::2, 0::2],
            x[..., 0::2, 1::2],
            x[..., 1::2, 1::2]
        ), axis=1)


class YoloDetect(DarknetLayer):
    def __init__(self, anchor: list, input_height: int, input_width: int,
                 scale_x_y: float = 1.0, new_coords: int = 0):
        super(YoloDetect, self).__init__()
        self.input_height = input_height
        self.input_width = input_width
        anchor = xp.array(anchor, dtype=xp.float32).reshape(1, -1, 1, 2)
        self.num_anchors = anchor.shape[1]
        self.add_weights('anchor', anchor)
        self.scale_x_y = scale_x_y
        self.new_coords = new_coords

    @staticmethod
    def gen_grid(h: int, w: int):
        rgx = xp.arange(h, dtype=xp.float32)
        rgy = xp.arange(w, dtype=xp.float32)
        yv, xv = xp.meshgrid(rgy, rgx, indexing='ij')
        grid = xp.stack((xv, yv), axis=-1).reshape(1, 1, -1, 2)
        return grid

    def __call__(self, x: array_t):
        b, c, h, w = x.shape
        assert self.input_height % h == 0
        assert self.input_width % w == 0
        assert self.input_height // h == self.input_width // w
        stride = self.input_height // h
        no = c // self.num_anchors

        x = x.reshape(b, self.num_anchors, no, h * w)
        x = xp.transpose(x, (0, 1, 3, 2))
        grid = self.gen_grid(h, w)

        if not self.new_coords:
            xy = xp.tanh(x[..., 0:2]) * (0.5 * self.scale_x_y * stride) + \
                 (grid + 0.5) * stride
            wh = xp.exp(x[..., 2:4]) * self.anchor
        else:
            xy = x[..., 0:2] * (self.scale_x_y * stride) + (grid - 0.5 * (self.scale_x_y - 1)) * stride
            wh = xp.square(x[..., 2:4]) * (self.anchor * 4)
        conf = act.sigmoid(x[..., 4:])
        output = xp.concatenate((xy, wh, conf), axis=-1)
        output = output.reshape(b, -1, no)
        return output


def _check_conv_nobn():
    import torch
    import cupy
    from torch import nn
    from torch.utils.dlpack import from_dlpack, to_dlpack
    from cupy._core.dlpack import toDlpack
    model = nn.Conv2d(3, 16, (5, 7), (3, 2), padding=0)
    model.weight.data.normal_()
    model.bias.data.normal_()
    model.cuda()
    model.eval()

    x = xp.random.rand(2, 3, 100, 111).astype(xp.float32)
    x_tensor = from_dlpack(toDlpack(x))

    with torch.no_grad():
        r1_tensor = model(x_tensor)
    r1 = cupy.from_dlpack(to_dlpack(r1_tensor))

    md_dark = Convolutional(3, 16, (5, 7), (3, 2), pad=False, use_bn=False, act=None)
    md_dark._set_weights_nobn(
        cupy.from_dlpack(to_dlpack(model.bias.data)),
        cupy.from_dlpack(to_dlpack(model.weight.data))
    )
    r2 = md_dark(x)
    print(xp.abs(r1 - r2).max())


def _check_conv():
    import torch
    import cupy
    from torch import nn
    from cupy._core.dlpack import toDlpack
    from torch.utils.dlpack import from_dlpack, to_dlpack
    model = nn.Sequential()
    model.conv = nn.Conv2d(3, 16, (5, 7), (3, 2), bias=False)
    model.bn = nn.BatchNorm2d(16)
    model.conv.weight.data.normal_()
    model.bn.weight.data.normal_()
    model.bn.bias.data.normal_()
    model.bn.running_mean.data.normal_()
    model.bn.running_var.data.uniform_(0.01, 1.)
    model.cuda()
    model.eval()

    x = xp.random.rand(2, 3, 100, 111).astype(xp.float32)
    x_tensor = from_dlpack(toDlpack(x))

    with torch.no_grad():
        r1_tensor = model(x_tensor)
    r1 = cupy.from_dlpack(to_dlpack(r1_tensor))

    md_dark = Convolutional(3, 16, (5, 7), (3, 2), pad=False, use_bn=True)
    md_dark._set_weights_bn(
        cupy.from_dlpack(to_dlpack(model.bn.bias.data)),
        cupy.from_dlpack(to_dlpack(model.bn.weight.data)),
        cupy.from_dlpack(to_dlpack(model.bn.running_mean.data)),
        cupy.from_dlpack(to_dlpack(model.bn.running_var.data)),
        cupy.from_dlpack(to_dlpack(model.conv.weight.data))
    )
    r2 = md_dark(x)
    print(r1.shape, r2.shape)
    print(xp.abs(r1 - r2).max())


if __name__ == '__main__':
    _check_conv_nobn()
    _check_conv()
