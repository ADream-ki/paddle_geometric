import warnings
from typing import Any, Callable, Optional, Union

import paddle

import paddle_geometric.typing


def is_compiling() -> bool:
    r"""Returns :obj:`True` in case PaddlePaddle is compiling via
    :meth:`paddle.jit.to_static`.
    """
    # if torch_geometric.typing.WITH_PT23:
    #     return torch.compiler.is_compiling()
    # if torch_geometric.typing.WITH_PT21:
    #     return torch._dynamo.is_compiling()
    return False  # pragma: no cover


def compile(
    model: Optional[paddle.nn.Layer] = None,
    *args: Any,
    **kwargs: Any,
) -> Union[paddle.nn.Layer, Callable[[paddle.nn.Layer], paddle.nn.Layer]]:
    r"""Optimizes the given :pyg:`PyG` model/function via
    :meth:`paddle.jit.to_static`.
    This function has the same signature as :meth:`paddle.jit.to_static`.

    Args:
        model: The model to compile.
        *args: Additional arguments of :meth:`paddle.jit.to_static`.
        **kwargs: Additional keyword arguments of :meth:`paddle.jit.to_static`.

    .. note::
        :meth:`paddle_geometric.compile` is deprecated in favor of
        :meth:`paddle.jit.to_static`.
    """
    warnings.warn("'paddle_geometric.compile' is deprecated in favor of "
                  "'paddle.jit.to_static'")
    if paddle_geometric.typing.WITH_PP31:
        return paddle.jit.to_static(model, *args, **kwargs)
    elif paddle_geometric.typing.WITH_PP30:
        return paddle.jit.to_static(model, backend='CINN', *args, **kwargs)
    else:
        raise RuntimeError

