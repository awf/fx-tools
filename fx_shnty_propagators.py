import operator
import numpy
import torch

from fx_shnty import (
    AbstractValue,
    AbstractTensor,
    fx_shape,
    fx_type,
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

    shapes = [fx_shape(a) for a in args]
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
    torch.sin,
    torch.cos,
    torch.neg,
):
    shnty_propagator_register(op, shnty_propagate_broadcast(op, None))

for method in ("neg", "sin", "cos"):
    shnty_propagator_register(
        (torch.Tensor, method), shnty_propagate_broadcast(op, None)
    )

# --------------  boolean ops
for op in (
    operator.lt,
    operator.le,
    operator.eq,
    operator.ne,
    operator.ge,
    operator.gt,
):
    shnty_propagator_register(op, shnty_propagate_broadcast(op, torch.bool))

# --------------  reshape


@shnty_propagator((torch.Tensor, "reshape"))
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
@shnty_propagator((torch.Tensor, "t"))
def _(x):
    if len(x.shape) != 2:
        raise NotImplementedError(f"check transpose implementation for snhty {x}")

    assert fx_is_tensor(x)
    sh = torch.Size((*x.shape[:-2], x.shape[-1], x.shape[-2]))
    return AbstractTensor(sh, x.dtype)


# --------------  @ matmul


@shnty_propagator((torch.Tensor, "matmul"))
@shnty_propagator(operator.matmul)
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


@shnty_propagator((torch.Tensor, "sum"))
@shnty_propagator(torch.sum)
def _(x, dim=None):
    assert fx_is_tensor(x)

    if not dim or len(dim) == 0:
        sh = torch.Size()  # sum over all elements, return type is scalar
    else:
        sh = torch.Size(s for i, s in enumerate(x.shape) if i not in dim)

    return AbstractTensor(sh, x.dtype)


@shnty_propagator(torch.transpose)
def _(x, m, n):
    assert fx_is_tensor(x)
    sh = list(x.shape)
    tmp = sh[m]
    sh[m] = sh[n]
    sh[n] = tmp
    return AbstractTensor(torch.Size(tuple(sh)), x.dtype)
