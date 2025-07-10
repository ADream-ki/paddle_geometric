from typing import Optional

import paddle
from paddle import Tensor

from paddle_geometric.experimental import disable_dynamic_shapes
from paddle_geometric.nn.aggr import Aggregation

# @finshed
class SortAggregation(Aggregation):
    r"""The pooling operator from the `"An End-to-End Deep Learning
    Architecture for Graph Classification"
    <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`_ paper,
    where node features are sorted in descending order based on their last
    feature channel. The first :math:`k` nodes form the output of the layer.

    .. note::

        :class:`SortAggregation` requires sorted indices :obj:`index` as input.
        Specifically, if you use this aggregation as part of
        :class:`~paddle_geometric.nn.conv.MessagePassing`, ensure that
        :obj:`edge_index` is sorted by destination nodes, either by manually
        sorting edge indices or by calling `Data.sort()`.

    Args:
        k (int): The number of nodes to hold for each graph.
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    @disable_dynamic_shapes(required_args=['dim_size', 'max_num_elements'])
    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ) -> Tensor:

        fill_value = x.detach().min() - 1
        batch_x, _ = self.to_dense_batch(x, index, ptr, dim_size, dim,
                                         fill_value=fill_value,
                                         max_num_elements=max_num_elements)
        B, N, D = tuple(batch_x.shape)

        perm = paddle.argsort(axis=-1, descending=True, x=batch_x[:, :, -1])
        arange = paddle.arange(B, dtype='int64') * N
        perm = perm + arange.view([-1, 1])

        batch_x = batch_x.view([B * N, D])
        batch_x = batch_x[perm]
        batch_x = batch_x.view([B, N, D])

        if N >= self.k:
            batch_x = batch_x[:, :self.k].contiguous()
        else:
            expand_batch_x = paddle.full(
                shape=(B, self.k - N, D), fill_value=fill_value, dtype=batch_x.dtype
            )
            batch_x = paddle.concat(x=[batch_x, expand_batch_x], axis=1)
        batch_x[batch_x == fill_value] = 0
        x = batch_x.view([B, self.k * D])

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(k={self.k})')
