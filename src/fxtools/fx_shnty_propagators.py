import operator
import numpy
import torch
from typing import List, Optional, Tuple

from .fx_shnty import (
    AbstractValue,
    AbstractTensor,
    fx_shnty_shape,
    fx_is_tensor,
    shnty_propagator,
    shnty_propagator_register,
)


# --------------  Broadcasting ops (e.g. add, mul, sin..)


def shnty_propagate_broadcast_aux(op, dty, *args):
    """
    Propagate OP, with normal Pytorch broadcasting semantics
    ARGS may be AbstractValue or values
    """
    msg = f"shnty_propagate_broadcast_aux {op}"

    shapes = [fx_shnty_shape(a) for a in args]
    sh = torch.broadcast_shapes(*shapes)

    # Type
    def get_dtype(a):
        if isinstance(a, (torch.Tensor, AbstractTensor)):
            return a.dtype
        if isinstance(a, AbstractValue):
            return a.ty
        return torch.tensor(a).dtype

    if not dty:
        dty = get_dtype(args[0])
        for arg in args:
            dty = torch.promote_types(dty, get_dtype(arg))

    return AbstractTensor(sh, dty)


def shnty_propagate_broadcast(op, dty):
    """
    Make a propagator for OP, with normal Pytorch broadcasting semantics
    """
    return lambda *args: shnty_propagate_broadcast_aux(op, dty, *args)


for op in (
    operator.mul,
    operator.truediv,
    operator.add,
    operator.neg,
    torch.ones_like,
    torch.relu,
    torch.atan2,
    torch.neg,
    torch.sin,
    torch.cos,
    torch.Tensor.neg,
    torch.Tensor.sin,
    torch.Tensor.cos,
):
    shnty_propagator_register(op, shnty_propagate_broadcast(op, None))

# --------------  boolean ops
for op, ty in (
    (operator.lt, torch.bool),
    (operator.le, torch.bool),
    (operator.eq, torch.bool),
    (operator.ne, torch.bool),
    (operator.ge, torch.bool),
    (operator.gt, torch.bool),
    (torch.Tensor.lt, torch.bool),
    (torch.Tensor.le, torch.bool),
    (torch.Tensor.eq, torch.bool),
    (torch.Tensor.ne, torch.bool),
    (torch.Tensor.ge, torch.bool),
    (torch.Tensor.gt, torch.bool),
    (torch.Tensor.long, torch.int64),
):
    shnty_propagator_register(op, shnty_propagate_broadcast(op, ty))

# --------------  reshape


@shnty_propagator(torch.Tensor.reshape)
def _(x, shape):
    assert fx_is_tensor(x)
    assert numpy.prod(shape) == numpy.prod(x.shape)
    return AbstractTensor(torch.Size(shape), x.dtype)


# --------------  torch.trace


@shnty_propagator(torch.trace)
def _(x):
    assert fx_is_tensor(x)
    if len(x.shape) != 2:
        raise NotImplementedError(
            f"check trace implementation for snhty {x}, it might be fine"
        )
    return AbstractTensor(x.shape[:-2], x.dtype)


# --------------  Tensor.t()


@shnty_propagator(torch.Tensor.T)
@shnty_propagator(torch.Tensor.t)
def _(x):
    if len(x.shape) != 2:
        raise NotImplementedError(f"check transpose implementation for snhty {x}")

    assert fx_is_tensor(x)
    sh = torch.Size((*x.shape[:-2], x.shape[-1], x.shape[-2]))
    return AbstractTensor(sh, x.dtype)


# --------------  transpose


@shnty_propagator(torch.transpose)
@shnty_propagator(torch.Tensor.transpose)
def _(x, m, n):
    assert fx_is_tensor(x)
    sh = list(x.shape)
    tmp = sh[m]
    sh[m] = sh[n]
    sh[n] = tmp
    return AbstractTensor(torch.Size(tuple(sh)), x.dtype)


# --------------  @ matmul


@shnty_propagator(operator.matmul)
@shnty_propagator(torch.Tensor.matmul)
def _(A, B):
    assert fx_is_tensor(A) and fx_is_tensor(B)
    # m x p times p x n
    m, pA = A.shape[-2:]
    pB, n = B.shape[-2:]
    leading_dims = A.shape[:-2]
    assert pA == pB
    assert leading_dims == B.shape[:-2]

    dty = torch.promote_types(A.dtype, B.dtype)
    return AbstractTensor(torch.Size((*leading_dims, m, n)), dty)


# --------------  torch.sum


@shnty_propagator(torch.sum)
@shnty_propagator(torch.Tensor.sum)
def _(x, dim=None):
    assert fx_is_tensor(x)

    if not dim or len(dim) == 0:
        sh = torch.Size()  # sum over all elements, return type is scalar
    else:
        sh = torch.Size(s for i, s in enumerate(x.shape) if i not in dim)

    return AbstractTensor(sh, x.dtype)


# --------------  torch.cumsum


@shnty_propagator(torch.cumsum)
@shnty_propagator(torch.Tensor.cumsum)
def _(x, dim=None):
    assert fx_is_tensor(x)
    return AbstractTensor(fx_shnty_shape(x), x.dtype)


# --------------  getitem


@shnty_propagator(operator.getitem)
def _(x, idx):
    assert fx_is_tensor(x)
    sh = fx_shnty_shape(x)
    assert len(idx) == len(sh)
    outsh = list(sh)
    for n, i in enumerate(idx):
        if isinstance(i, slice):
            if i.start == None and i.stop == None:
                outsh[n] = sh[n]
            else:
                raise NotImplementedError("!")
        elif isinstance(i, int):
            outsh[n] = -1
        else:
            raise NotImplementedError("!")

    outsh = [i for i in outsh if i != -1]

    return AbstractTensor(torch.Size(outsh), x.dtype)


#####
##### Modules
#####


@shnty_propagator(torch.nn.Dropout)
@shnty_propagator(torch.nn.LayerNorm)
def _(mod, arg):
    return AbstractTensor(fx_shnty_shape(arg), arg.dtype)


@shnty_propagator(torch.nn.functional.layer_norm)
def layer_norm(
    input: AbstractTensor,
    normalized_shape: List[int],
    weight: Optional[AbstractTensor] = None,
    bias: Optional[AbstractTensor] = None,
    eps: float = 1e-5,
) -> AbstractTensor:
    assert not weight and not bias
    return input


@shnty_propagator(torch.nn.functional.dropout)
def _(
    input: AbstractTensor, p: float = 0.5, training: bool = True, inplace: bool = False
) -> AbstractTensor:
    return input


@shnty_propagator(torch.nn.TransformerEncoder)
def _(mod, arg):
    return AbstractTensor(fx_shnty_shape(arg), arg.dtype)


@shnty_propagator(torch.nn.Embedding)
def _(module, input):
    assert fx_is_tensor(input)
    assert input.dtype == torch.int64
    sh = fx_shnty_shape(input)
    sh = torch.Size((*sh, module.embedding_dim))
    print(
        "Warning: shnty_propagator(torch.nn.modules.sparse.Embedding): assuming float32"
    )
    return AbstractTensor(sh, torch.float32)


@shnty_propagator(torch.nn.functional.embedding)
def _(input, weight):
    assert fx_is_tensor(weight)
    assert fx_is_tensor(input)
    assert input.dtype == torch.int64
    sh = fx_shnty_shape(input)
    sh = torch.Size((*sh, weight.shape[1]))
    return AbstractTensor(sh, weight.dtype)


# @shnty_propagator(torch.nn.functional.multi_head_attention_forward)
# def _(
#     query: AbstractTensor,
#     key: AbstractTensor,
#     value: AbstractTensor,
#     embed_dim_to_check: int,
#     num_heads: int,
#     in_proj_weight: Optional[AbstractTensor],
#     in_proj_bias: Optional[AbstractTensor],
#     bias_k: Optional[AbstractTensor],
#     bias_v: Optional[AbstractTensor],
#     add_zero_attn: bool,
#     dropout_p: float,
#     out_proj_weight: AbstractTensor,
#     out_proj_bias: Optional[AbstractTensor],
#     training: bool = True,
#     key_padding_mask: Optional[AbstractTensor] = None,
#     need_weights: bool = True,
#     attn_mask: Optional[AbstractTensor] = None,
#     use_separate_proj_weight: bool = False,
#     q_proj_weight: Optional[AbstractTensor] = None,
#     k_proj_weight: Optional[AbstractTensor] = None,
#     v_proj_weight: Optional[AbstractTensor] = None,
#     static_k: Optional[AbstractTensor] = None,
#     static_v: Optional[AbstractTensor] = None,
#     average_attn_weights: bool = True,
# ) -> Tuple[AbstractTensor, Optional[AbstractTensor]]:


@shnty_propagator(torch.nn.Linear)
def _(mod, arg):
    assert fx_is_tensor(arg)
    assert mod.in_features == arg.shape[-1]
    sh = torch.Size((*arg.shape[:-1], mod.out_features))
    return AbstractTensor(sh, arg.dtype)


@shnty_propagator(torch.nn.ReLU)
def _(mod, arg):
    return arg
