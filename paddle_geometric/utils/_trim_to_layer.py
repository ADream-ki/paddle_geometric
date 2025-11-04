from typing import Dict, List, Optional, Tuple, Union, overload

import paddle
from paddle import Tensor

from paddle_geometric import EdgeIndex
from paddle_geometric.typing import (
    Adj,
    EdgeType,
    MaybeHeteroAdjTensor,
    MaybeHeteroEdgeTensor,
    MaybeHeteroNodeTensor,
    NodeType,
    SparseStorage,
    SparseTensor,
)


@overload
def trim_to_layer(
    layer: int,
    num_sampled_nodes_per_hop: List[int],
    num_sampled_edges_per_hop: List[int],
    x: Tensor,
    edge_index: Adj,
    edge_attr: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    pass


@overload
def trim_to_layer(
    layer: int,
    num_sampled_nodes_per_hop: Dict[NodeType, List[int]],
    num_sampled_edges_per_hop: Dict[EdgeType, List[int]],
    x: Dict[NodeType, Tensor],
    edge_index: Dict[EdgeType, Adj],
    edge_attr: Optional[Dict[EdgeType, Tensor]] = None,
) -> Tuple[Dict[NodeType, Tensor], Dict[EdgeType, Adj], Optional[Dict[
        EdgeType, Tensor]]]:
    pass


def trim_to_layer(
    layer: int,
    num_sampled_nodes_per_hop: Union[List[int], Dict[NodeType, List[int]]],
    num_sampled_edges_per_hop: Union[List[int], Dict[EdgeType, List[int]]],
    x: MaybeHeteroNodeTensor,
    edge_index: MaybeHeteroEdgeTensor,
    edge_attr: Optional[MaybeHeteroEdgeTensor] = None,
) -> Tuple[MaybeHeteroNodeTensor, MaybeHeteroAdjTensor,
           Optional[MaybeHeteroEdgeTensor]]:
    if layer <= 0:
        return x, edge_index, edge_attr
    if isinstance(num_sampled_edges_per_hop, dict):
        assert isinstance(num_sampled_nodes_per_hop, dict)
        assert isinstance(x, dict)
        x = {
            k: trim_feat(v, layer, num_sampled_nodes_per_hop[k])
            for k, v in x.items()
        }
        assert isinstance(edge_index, dict)
        edge_index = {
            k:
            trim_adj(
                v,
                layer,
                num_sampled_nodes_per_hop[k[0]],
                num_sampled_nodes_per_hop[k[-1]],
                num_sampled_edges_per_hop[k],
            )
            for k, v in edge_index.items()
        }
        if edge_attr is not None:
            assert isinstance(edge_attr, dict)
            edge_attr = {
                k: trim_feat(v, layer, num_sampled_edges_per_hop[k])
                for k, v in edge_attr.items()
            }
        return x, edge_index, edge_attr
    assert isinstance(num_sampled_nodes_per_hop, list)
    assert isinstance(x, paddle.Tensor)
    x = trim_feat(x, layer, num_sampled_nodes_per_hop)
    assert isinstance(edge_index, (Tensor, SparseTensor))
    edge_index = trim_adj(
        edge_index,
        layer,
        num_sampled_nodes_per_hop,
        num_sampled_nodes_per_hop,
        num_sampled_edges_per_hop,
    )
    if edge_attr is not None:
        assert isinstance(edge_attr, paddle.Tensor)
        edge_attr = trim_feat(edge_attr, layer, num_sampled_edges_per_hop)
    return x, edge_index, edge_attr


class TrimToLayer(paddle.nn.Layer):
    def forward(
        self,
        layer: int,
        num_sampled_nodes_per_hop: Optional[List[int]],
        num_sampled_edges_per_hop: Optional[List[int]],
        x: Tensor,
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Adj, Optional[Tensor]]:
        if not isinstance(num_sampled_nodes_per_hop, list) and isinstance(
                num_sampled_edges_per_hop, list):
            raise ValueError("'num_sampled_nodes_per_hop' needs to be given")
        if not isinstance(num_sampled_edges_per_hop, list) and isinstance(
                num_sampled_nodes_per_hop, list):
            raise ValueError("'num_sampled_edges_per_hop' needs to be given")
        if num_sampled_nodes_per_hop is None:
            return x, edge_index, edge_attr
        if num_sampled_edges_per_hop is None:
            return x, edge_index, edge_attr
        return trim_to_layer(
            layer,
            num_sampled_nodes_per_hop,
            num_sampled_edges_per_hop,
            x,
            edge_index,
            edge_attr,
        )


def trim_feat(x: Tensor, layer: int, num_samples_per_hop: List[int]) -> Tensor:
    if layer <= 0:
        return x
    start_3 = x.shape[0] + 0 if 0 < 0 else 0
    return paddle.slice(x, [0], [start_3],
                        [start_3 + (x.shape[0] - num_samples_per_hop[-layer])])


def trim_adj(
    edge_index: Adj,
    layer: int,
    num_sampled_src_nodes_per_hop: List[int],
    num_sampled_dst_nodes_per_hop: List[int],
    num_sampled_edges_per_hop: List[int],
) -> Adj:
    if layer <= 0:
        return edge_index
    if isinstance(edge_index, paddle.Tensor):
        start_4 = edge_index.shape[1] + 0 if 0 < 0 else 0
        edge_index = paddle.slice(
            edge_index,
            [1],
            [start_4],
            [
                start_4 +
                (edge_index.shape[1] - num_sampled_edges_per_hop[-layer])
            ],
        )
        if isinstance(edge_index, EdgeIndex):
            num_rows, num_cols = edge_index.sparse_size()
            if num_rows is not None:
                num_rows -= num_sampled_src_nodes_per_hop[-layer]
            if num_cols is not None:
                num_cols -= num_sampled_dst_nodes_per_hop[-layer]
            edge_index.sparse_resize_(num_rows, num_cols)
        return edge_index
    elif isinstance(edge_index, SparseTensor):
        size = (
            edge_index.size(0) - num_sampled_dst_nodes_per_hop[-layer],
            edge_index.size(1) - num_sampled_src_nodes_per_hop[-layer],
        )
        num_seed_nodes = size[0] - num_sampled_dst_nodes_per_hop[-(layer + 1)]
        return trim_sparse_tensor(edge_index, size, num_seed_nodes)
    raise ValueError(f"Unsupported 'edge_index' type '{type(edge_index)}'")


def trim_sparse_tensor(src: SparseTensor, size: Tuple[int, int],
                       num_seed_nodes: int) -> SparseTensor:
    rowptr, col, value = src.csr()
    start_5 = rowptr.shape[0] + 0 if 0 < 0 else 0
    rowptr = paddle.slice(rowptr, [0], [start_5],
                          [start_5 + size[0] + 1]).clone()
    rowptr[num_seed_nodes + 1:] = rowptr[num_seed_nodes]
    start_6 = col.shape[0] + 0 if 0 < 0 else 0
    col = paddle.slice(col, [0], [start_6], [start_6 + rowptr[-1]])
    if value is not None:
        start_7 = value.shape[0] + 0 if 0 < 0 else 0
        value = paddle.slice(value, [0], [start_7], [start_7 + rowptr[-1]])
    csr2csc = src.storage._csr2csc
    if csr2csc is not None:
        csr2csc = csr2csc[csr2csc < len(col)]
    storage = SparseStorage(
        row=None,
        rowptr=rowptr,
        col=col,
        value=value,
        sparse_sizes=size,
        rowcount=None,
        colptr=None,
        colcount=None,
        csr2csc=csr2csc,
        csc2csr=None,
        is_sorted=True,
        trust_data=True,
    )
    return src.from_storage(storage)
