import math
from typing import Optional

import paddle
from paddle import Tensor
from paddle_geometric.nn.aggr import Aggregation
from paddle_geometric.utils import softmax

# @finshed
class SumAggregation(Aggregation):
    r"""An aggregation operator that sums up features across a set of elements."""
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='sum')

# @finshed
class MeanAggregation(Aggregation):
    r"""An aggregation operator that averages features across a set of elements."""
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='mean')

# @finshed
class MaxAggregation(Aggregation):
    r"""An aggregation operator that takes the feature-wise maximum across a set of elements."""
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='max')

# @finshed
class MinAggregation(Aggregation):
    r"""An aggregation operator that takes the feature-wise minimum across a set of elements."""
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        return self.reduce(x, index, ptr, dim_size, dim, reduce='min')

# @finshed
class MulAggregation(Aggregation):
    r"""An aggregation operator that multiplies features across a set of elements."""
    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        self.assert_index_present(index)
        return self.reduce(x, index, None, dim_size, dim, reduce='mul')

# @finshed
class VarAggregation(Aggregation):
    r"""An aggregation operator that calculates the feature-wise variance across a set of elements."""
    def __init__(self, semi_grad: bool = False):
        super().__init__()
        self.semi_grad = semi_grad

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        mean = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
        if self.semi_grad:
            with paddle.no_grad():
                mean2 = self.reduce(x * x, index, ptr, dim_size, dim, 'mean')
        else:
            mean2 = self.reduce(x * x, index, ptr, dim_size, dim, 'mean')
        return mean2 - mean * mean

# @finshed
class StdAggregation(Aggregation):
    r"""An aggregation operator that calculates the feature-wise standard deviation across a set of elements."""
    def __init__(self, semi_grad: bool = False):
        super().__init__()
        self.var_aggr = VarAggregation(semi_grad)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        var = self.var_aggr(x, index, ptr, dim_size, dim)
        out = var.clip(min=1e-5).sqrt()
        out = out.masked_fill(out <= math.sqrt(1e-5), 0.0)
        return out

# @finshed
class SoftmaxAggregation(Aggregation):
    r"""The softmax aggregation operator based on a temperature term."""
    def __init__(self, t: float = 1.0, learn: bool = False,
                 semi_grad: bool = False, channels: int = 1):
        super().__init__()

        if learn and semi_grad:
            raise ValueError(
                f"Cannot enable 'semi_grad' in '{self.__class__.__name__}' in "
                f"case the temperature term 't' is learnable")

        if not learn and channels != 1:
            raise ValueError(f"Cannot set 'channels' greater than '1' in case "
                             f"'{self.__class__.__name__}' is not trainable")

        self._init_t = t
        self.learn = learn
        self.semi_grad = semi_grad
        self.channels = channels

        self.t = paddle.create_parameter(shape=[channels], dtype='float32') if learn else t
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.t, Tensor):
            self.t.set_value(paddle.full_like(self.t, self._init_t))

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        t = self.t
        if self.channels != 1:
            self.assert_two_dimensional_input(x, dim)
            assert isinstance(t, Tensor)
            t = t.view([-1, self.channels])

        alpha = x
        if not isinstance(t, (int, float)) or t != 1:
            alpha = x * t

        if not self.learn and self.semi_grad:
            with paddle.no_grad():
                alpha = softmax(alpha, index, ptr, dim_size, dim)
        else:
            alpha = softmax(alpha, index, ptr, dim_size, dim)
        return self.reduce(x * alpha, index, ptr, dim_size, dim, reduce='sum')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(learn={self.learn})')

# @finshed
class PowerMeanAggregation(Aggregation):
    r"""The powermean aggregation operator based on a power term."""
    def __init__(self, p: float = 1.0, learn: bool = False, channels: int = 1):
        super().__init__()

        if not learn and channels != 1:
            raise ValueError(f"Cannot set 'channels' greater than '1' in case "
                             f"'{self.__class__.__name__}' is not trainable")

        self._init_p = p
        self.learn = learn
        self.channels = channels

        self.p = paddle.create_parameter(shape=[channels], dtype='float32') if learn else p
        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.p, Tensor):
            self.p.set_value(paddle.full_like(self.p, self._init_p))

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        p = self.p
        if self.channels != 1:
            assert isinstance(p, Tensor)
            self.assert_two_dimensional_input(x, dim)
            p = p.view([-1, self.channels])

        if not isinstance(p, (int, float)) or p != 1:
            x = x.clip(min=0, max=100).pow(p)

        out = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')

        if not isinstance(p, (int, float)) or p != 1:
            out = out.clip(min=0, max=100).pow(1. / p)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(learn={self.learn})')
