from typing import List, Optional, Union
import paddle
from paddle import Tensor

from paddle_geometric.nn.aggr import Aggregation
from paddle_geometric.utils import cumsum

# @finshed
class QuantileAggregation(Aggregation):
    r"""An aggregation operator that returns the feature-wise :math:`q`-th
    quantile of a set :math:`\mathcal{X}`.

    Args:
        q (float or list): The quantile value(s) :math:`q`. Can be a scalar or
            a list of scalars in the range :math:`[0, 1]`. If more than a
            quantile is passed, the results are concatenated.
        interpolation (str): Interpolation method applied if the quantile point
            :math:`q\cdot n` lies between two values
            :math:`a \le b`. Can be one of the following:

            * :obj:`"lower"`: Returns the one with lowest value.
            * :obj:`"higher"`: Returns the one with highest value.
            * :obj:`"midpoint"`: Returns the average of the two values.
            * :obj:`"nearest"`: Returns the one whose index is nearest to the
              quantile point.
            * :obj:`"linear"`: Returns a linear combination of the two
              elements, defined as :math:`f(a, b) = a + (b - a)\cdot(q\cdot n - i)`.
    """
    interpolations = {'linear', 'lower', 'higher', 'nearest', 'midpoint'}

    def __init__(self, q: Union[float, List[float]],
                 interpolation: str = 'linear', fill_value: float = 0.0):
        super().__init__()

        qs = [q] if not isinstance(q, (list, tuple)) else q
        if len(qs) == 0:
            raise ValueError("Provide at least one quantile value for `q`.")
        if not all(0. <= quantile <= 1. for quantile in qs):
            raise ValueError("`q` must be in the range [0, 1].")
        if interpolation not in self.interpolations:
            raise ValueError(f"Invalid interpolation method "
                             f"got ('{interpolation}')")

        self._q = q
        self.register_buffer('q', paddle.to_tensor(qs).view([-1, 1]))
        self.interpolation = interpolation
        self.fill_value = fill_value

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        dim = x.ndim + dim if dim < 0 else dim

        self.assert_index_present(index)
        assert index is not None  # Required for TorchScript.

        count = paddle.bincount(index, minlength=dim_size or 0)
        ptr = cumsum(count)[:-1]

        # In case there exists dangling indices (`dim_size > index.max()`), we
        # need to clamp them to prevent out-of-bound issues:
        if dim_size is not None:
            ptr = paddle.clip(ptr, max=x.shape[dim] - 1)

        q_point = self.q * (count - 1) + ptr
        q_point = q_point.t().reshape([-1])

        shape = [1] * x.ndim
        shape[dim] = -1
        index = index.view(shape).expand_as(x)

        # Two sorts: the first one on the value,
        # the second (stable) on the indices:
        x, x_perm = paddle.sort(x=x, axis=dim), paddle.argsort(x=x, axis=dim)
        index = index.take_along_axis(x_perm, axis=dim)
        index, index_perm = paddle.sort(x=index, axis=dim, stable=True), paddle.argsort(
            x=index, axis=dim, stable=True
        )
        x = x.take_along_axis(index_perm, axis=dim)

        # Compute the quantile interpolations:
        if self.interpolation == "lower":
            quantile = x.index_select(
                axis=dim, index=q_point.floor().astype(dtype="int64")
            )
        elif self.interpolation == "higher":
            quantile = x.index_select(
                axis=dim, index=q_point.ceil().astype(dtype="int64")
            )
        elif self.interpolation == "nearest":
            quantile = x.index_select(
                axis=dim, index=q_point.round().astype(dtype="int64")
            )
        else:
            l_quant = x.index_select(
                axis=dim, index=q_point.floor().astype(dtype="int64")
            )
            r_quant = x.index_select(
                axis=dim, index=q_point.ceil().astype(dtype="int64")
            )
            if self.interpolation == "linear":
                q_frac = q_point.frac().view(shape)
                quantile = l_quant + (r_quant - l_quant) * q_frac
            else:
                quantile = 0.5 * l_quant + 0.5 * r_quant

        repeats = self.q.size
        mask = (count == 0).repeat_interleave(repeats=repeats).view(shape)
        out = quantile.masked_fill(mask=mask, value=self.fill_value)

        if self.q.size > 1:
            shape = list(tuple(out.shape))
            shape = shape[:dim] + [shape[dim] // self.q.size, -1] + shape[dim + 2 :]
            out = out.view(shape)

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(q={self._q})'

# @finshed
class MedianAggregation(QuantileAggregation):
    r"""An aggregation operator that returns the feature-wise median of a set.

    Args:
        fill_value (float, optional): The default value in the case no entry is
            found for a given index (default: :obj:`0.0`).
    """
    def __init__(self, fill_value: float = 0.0):
        super().__init__(0.5, 'lower', fill_value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
