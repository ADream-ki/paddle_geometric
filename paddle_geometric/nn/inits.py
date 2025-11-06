import math
import warnings
from typing import Any

import paddle


def uniform(size: int, value: Any):
    if isinstance(value, paddle.Tensor):
        bound = 1.0 / math.sqrt(size)
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, "parameters") else []:
            uniform(size, v)
        for v in value.buffers() if hasattr(value, "buffers") else []:
            uniform(size, v)


def kaiming_uniform(value: Any, fan: int, a: float):
    if isinstance(value, paddle.Tensor):
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        value.data.uniform_(-bound, bound)
    else:
        for v in value.parameters() if hasattr(value, "parameters") else []:
            kaiming_uniform(v, fan, a)
        for v in value.buffers() if hasattr(value, "buffers") else []:
            kaiming_uniform(v, fan, a)


def glorot(value: Any):
    if isinstance(value, paddle.Tensor):
        stdv = math.sqrt(6.0 / (value.shape[-2] + value.shape[-1]))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, "parameters") else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, "buffers") else []:
            glorot(v)


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        paddle.nn.init.orthogonal_(tensor.data)
        scale /= (tensor.shape[-2] + tensor.shape[-1]) * tensor.var()
        tensor.data *= scale.sqrt()


def constant(value: Any, fill_value: float):
    if isinstance(value, paddle.Tensor):
        value.data.fill_(fill_value)
    else:
        for v in value.parameters() if hasattr(value, "parameters") else []:
            constant(v, fill_value)
        for v in value.buffers() if hasattr(value, "buffers") else []:
            constant(v, fill_value)


def zeros(value: Any):
    constant(value, 0.)


def ones(tensor: Any):
    constant(tensor, 1.)


def normal(value: Any, mean: float, std: float):
    if isinstance(value, paddle.Tensor):
        value.data.normal_(mean, std)
    else:
        for v in value.parameters() if hasattr(value, "parameters") else []:
            normal(v, mean, std)
        for v in value.buffers() if hasattr(value, "buffers") else []:
            normal(v, mean, std)


def reset(value: Any):
    warnings.warn(
        "PaddlePaddle does not implement 'reset_parameters'; "
        "the 'reset' function will have no effect.",
        UserWarning,
        stacklevel=2,
    )

    if hasattr(value, "reset_parameters"):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, "children") else []:
            reset(child)
