import paddle
import pytest

from paddle_geometric import place2devicestr
from paddle_geometric.paddle_utils import *  # noqa
from paddle_geometric.testing import withCUDA, withDevice, withPackage
from paddle_geometric.utils import group_argsort, group_cat, scatter
from paddle_geometric.utils._scatter import scatter_argmax


def test_scatter_validate():
    src = paddle.randn(shape=[100, 32])
    index = paddle.randint(low=0, high=10, shape=(100, ), dtype="int64")

    with pytest.raises(ValueError, match="must be one-dimensional"):
        scatter(src, index.view(-1, 1))

    with pytest.raises(ValueError, match="must lay between 0 and 1"):
        scatter(src, index, dim=2)

    with pytest.raises(ValueError, match="invalid `reduce` argument 'std'"):
        scatter(src, index, reduce="std")


@withDevice
@withPackage("paddle_scatter")
@pytest.mark.parametrize("reduce", ["sum", "add", "mean", "min", "max"])
def test_scatter(reduce, device):
    import paddle_scatter

    src = paddle.randn(shape=[100, 16], device=device)
    index = paddle.randint(low=0, high=8, shape=(100, ))
    index = index.to(device=device)

    # if device.type == "mps" and reduce in ["min", "max"]:
    #     with pytest.raises(NotImplementedError, match="for the MPS device"):
    #         scatter(src, index, dim=0, reduce=reduce)
    #     return

    out1 = scatter(src, index, dim=0, reduce=reduce)
    out2 = paddle_scatter.scatter(src, index, dim=0, reduce=reduce)
    assert place2devicestr(out1.place) == device
    assert paddle.allclose(x=out1, y=out2, atol=1e-06).item()

    # jit = paddle.jit.to_static(function=scatter)
    # out3 = jit(src, index, dim=0, reduce=reduce)
    # assert paddle.allclose(x=out1, y=out3, atol=1e-06).item()

    # # src = paddle.randn(shape=[8, 100, 16])
    # # out1 = scatter(src, index, dim=1, reduce=reduce)
    # # out2 = paddle_scatter.scatter(src, index, dim=1, reduce=reduce)
    # # assert out1.place == device
    # # assert paddle.allclose(x=out1, y=out2, atol=1e-06).item()


@withDevice
@pytest.mark.parametrize("reduce", ["sum", "add", "mean", "min", "max"])
def test_scatter_backward(reduce, device):
    out_25 = paddle.randn(shape=[8, 100, 16], device=device)
    out_25.stop_gradient = not True
    src = out_25
    index = paddle.randint(low=0, high=8, shape=(100, ))
    index = index.to(device=device)
    # if device.type == "mps" and reduce in ["min", "max"]:
    #     with pytest.raises(NotImplementedError, match="for the MPS device"):
    #         scatter(src, index, dim=1, reduce=reduce)
    #     return
    out = scatter(src, index, dim=1, reduce=reduce)
    assert src.grad is None
    out.mean().backward()
    assert src.grad is not None


@withDevice
def test_scatter_any(device):
    src = paddle.randn(shape=[6, 4], device=device)
    index = paddle.to_tensor(data=[0, 0, 1, 1, 2, 2], place=device)

    out = scatter(src, index, dim=0, reduce="any")

    for i in range(3):
        for j in range(4):
            assert float(out[i, j]) in src[2 * i:2 * i + 2, j].tolist()


@withDevice
@pytest.mark.parametrize("num_groups", [4])
@pytest.mark.parametrize("descending", [False, True])
def test_group_argsort(num_groups, descending, device):
    src = paddle.randn(shape=[20], device=device)
    index = paddle.randint(low=0, high=num_groups, shape=(20, ))
    index = index.to(device=device)

    out = group_argsort(src, index, 0, num_groups, descending=descending)

    expected = paddle.empty_like(x=index)
    for i in range(num_groups):
        mask = index == i
        tmp = src[mask].argsort(descending=descending)
        perm = paddle.empty_like(x=tmp)
        perm[tmp] = paddle.arange(end=tmp.size)
        expected[mask] = perm

    assert paddle.equal_all(x=out, y=expected).item()
    empty_tensor = paddle.to_tensor(data=[], place=device)

    out = group_argsort(empty_tensor, empty_tensor)
    assert out.numel() == 0


@withCUDA
@withPackage("paddle_scatter")
def test_scatter_argmax(device):
    src = paddle.arange(end=5, device=device)
    index = paddle.to_tensor(data=[2, 2, 0, 0, 3], place=device)

    # old_state = paddle_geometric.typing.WITH_PADDLE_SCATTER
    # paddle_geometric.typing.WITH_PADDLE_SCATTER = False
    argmax = scatter_argmax(src, index, dim_size=6)
    # paddle_geometric.typing.WITH_PADDLE_SCATTER = old_state
    assert argmax.tolist() == [3, 5, 1, 4, 5, 5]


@withDevice
def test_group_cat(device):
    x1 = paddle.randn(shape=[4, 4], device=device)
    x2 = paddle.randn(shape=[2, 4], device=device)
    index1 = paddle.to_tensor(data=[0, 0, 1, 2], place=device)
    index2 = paddle.to_tensor(data=[0, 2], place=device)

    expected = paddle.concat(x=[x1[:2], x2[:1], x1[2:4], x2[1:]], axis=0)

    out, index = group_cat([x1, x2], [index1, index2], dim=0,
                           return_index=True)

    assert paddle.equal_all(x=out, y=expected).item()
    assert index.tolist() == [0, 0, 0, 1, 2, 2]
