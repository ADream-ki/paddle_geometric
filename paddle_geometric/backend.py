from typing import Optional

import paddle

# If set to `True`, PyG is configured to use the `segment_matmul` and
# `grouped_matmul` kernels from `pyg-lib` to parallelize matrix multiplication
# across segments/groups of potentially varying size.
# If set to `None`, will automatically decide whether to utilize
# `segment_matmul` and `grouped_matmul` based on input sizes.
# Requires `pyg-lib` to be installed.
use_segment_matmul: Optional[bool] = None

# Helper functions ############################################################


def use_segment_matmul_heuristic(
    num_segments: int,
    max_segment_size: int,
    in_channels: int,
    out_channels: int,
) -> bool:
    r"""A heuristic based on input sizes to determine whether the usage of
    :meth:`segment_matmul` can speed up computation.
    """
    # NOTE This heuristic was learned on an A100 via sklearn using a simple
    # StandardScaler() -> LinearSVC() model.
    # For now, it is only used in combination with `RGCNConv`.
    x = paddle.to_tensor(
        data=[num_segments, max_segment_size, in_channels, out_channels]
    ).cast('float32')
    mean = paddle.to_tensor(
        data=[125.11603189, 12133.21523472, 163.81222321, 32.43755536]
    )
    std = paddle.to_tensor(
        data=[163.34480422, 27572.94543809, 177.6426489, 56.82103934]
    )
    weight = paddle.to_tensor(
        data=[2.43877659, 1.67583047, -0.000520527282, 0.343925501]
    )
    bias = 1.20236999
    x = (x - mean) / std

    return bool(paddle.matmul(x, weight) >= bias)
