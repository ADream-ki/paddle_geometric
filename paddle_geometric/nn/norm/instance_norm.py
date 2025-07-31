from typing import Optional

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer

from paddle_geometric.typing import OptTensor
from paddle_geometric.utils import degree, scatter
from typing import Literal
from paddle._typing import (
    ParamAttrLike,
)
from paddle.nn.initializer import Constant

class _InstanceNormBase(Layer):
    """
    This class is based class for InstanceNorm1D, 2d, 3d.

    See InstanceNorm1D, InstanceNorm2D or InstanceNorm3D for more details.
    """

    scale: Tensor | None
    bias: Tensor | None

    def __init__(
        self,
        num_features: int,
        epsilon: float = 1e-5,
        momentum: float = 0.9,
        weight_attr: ParamAttrLike | None = None,
        bias_attr: ParamAttrLike | None = None,
        data_format: Literal["NCHW"] = "NCHW",
        name: str | None = None,
    ) -> None:
        super().__init__()

        if weight_attr is False or bias_attr is False:
            assert (
                weight_attr == bias_attr
            ), "weight_attr and bias_attr must be set to False at the same time in InstanceNorm"
        self._momentum = momentum
        self._epsilon = epsilon
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._num_features = num_features
        self._data_format = data_format

        if weight_attr is not False and bias_attr is not False:
            self.scale = self.create_parameter(
                attr=self._weight_attr,
                shape=[num_features],
                default_initializer=Constant(1.0),
                is_bias=False,
            )
            self.bias = self.create_parameter(
                attr=self._bias_attr,
                shape=[num_features],
                default_initializer=Constant(0.0),
                is_bias=True,
            )
        else:
            self.scale = None
            self.bias = None

    def _check_input_dim(self, input):
        raise NotImplementedError("InstanceNorm Base error")

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        return F.instance_norm(
            input,
            weight=self.scale,
            bias=self.bias,
            momentum=self._momentum,
            eps=self._epsilon,
            data_format=self._data_format,
        )

    def extra_repr(self) -> str:
        return f'num_features={self._num_features}, epsilon={self._epsilon}'

# @finshed
class InstanceNorm(_InstanceNormBase):
    r"""Applies instance normalization over each individual example in a batch
    of node features as described in the `"Instance Normalization: The Missing
    Ingredient for Fast Stylization" <https://arxiv.org/abs/1607.08022>`_
    paper.

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    The mean and standard-deviation are calculated per-dimension separately for
    each object in a mini-batch.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`False`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses instance statistics in both training and eval modes.
            (default: :obj:`False`)
    """
    def __init__(
        self,
        in_channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ):
        assert track_running_stats is False, "Not supported yet."
        super().__init__(
            in_channels, eps, 1-momentum, weight_attr=affine, bias_attr=affine
        )

    def reset_parameters(self):
        """Resets all learnable parameters of the module."""
        pass
        # paddle.nn.initializer.Constant(1.0)(self.weight)
        # paddle.nn.initializer.Constant(0.0)(self.bias)
        # paddle.nn.initializer.Constant(0.0)(self.running_mean)
        # paddle.nn.initializer.Constant(1.0)(self.running_var)

    def forward(self, x: Tensor, batch: OptTensor = None,
                batch_size: Optional[int] = None) -> Tensor:
        r"""Forward pass.

        Args:
            x (paddle.Tensor): The source tensor.
            batch (paddle.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example. (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given. (default: :obj:`None`)
        """
        if batch is None:
            out = paddle.nn.functional.instance_norm(
                x=x.t().unsqueeze(axis=0),
                running_mean=self.running_mean,
                running_var=self.running_var,
                weight=self.scale,
                bias=self.bias,
                # use_input_stats=self.training or not self.track_running_stats,
                eps=self.eps,
                momentum=self.momentum,
            )
            return out.squeeze(axis=0).t()

        if batch_size is None:
            batch_size = int(batch.max()) + 1

        mean = var = unbiased_var = x  # Dummies.

        if self.training or not self.track_running_stats:
            norm = degree(batch, batch_size, dtype=x.dtype).clip_(min=1)
            norm = norm.view([-1, 1])
            unbiased_norm = (norm - 1).clip_(min=1)

            mean = scatter(x, batch, dim=0, dim_size=batch_size, reduce='sum') / norm

            x = x - mean.index_select(0, batch)

            var = scatter(x * x, batch, dim=0, dim_size=batch_size, reduce='sum')
            unbiased_var = var / unbiased_norm
            var = var / norm

            momentum = self.momentum
            if self.running_mean is not None:
                self.running_mean = (
                    1 - momentum) * self.running_mean + momentum * mean.mean(0)
            if self.running_var is not None:
                self.running_var = (
                    1 - momentum
                ) * self.running_var + momentum * unbiased_var.mean(0)
        else:
            if self.running_mean is not None:
                mean = self.running_mean.view([-1, 1]).expand([batch_size, -1])
            if self.running_var is not None:
                var = self.running_var.view([-1, 1]).expand([batch_size, -1])

            x = x - mean.index_select(0, batch)

        out = x / (var + self.eps).sqrt().index_select(0, batch)

        if self.scale is not None and self.bias is not None:
            out = out * self.scale.view([-1, 1]) + self.bias.view([-1, 1])

        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_features})'
