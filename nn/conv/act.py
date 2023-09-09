from nn.conv.utils import array_t
from nn.conv.utils import get_array_module


def leaky_relu(x: array_t, negative_slope: float = 0.1) -> array_t:
    y = x.copy()
    y[y < 0] *= negative_slope
    return y


def sigmoid(x: array_t) -> array_t:
    xp = get_array_module(x)
    return xp.tanh(x) * 0.5 + 0.5


def mish(x: array_t) -> array_t:
    xp = get_array_module(x)
    softplus_x = xp.log(1 + xp.exp(x))
    return x * xp.tanh(softplus_x)
