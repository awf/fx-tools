import operator
import torch

from fx_shnty import (
    AbstractValue,
    AbstractTensor,
    fx_is_tensor,
    shnty_propagator,
    shnty_propagator_register,
)

# --------------  Broadcasting ops (e.g. add, mul, sin..)


def shnty_propagate_broadcast_aux(op, *args):
    """
    Propagate OP, with normal Pytorch broadcasting semantics
    ARGS may be AbstractValue or values
    """
    msg = f"shnty_propagate_broadcast_aux {op}"

    # Shape
    def get_shape(a):
        if isinstance(a, (AbstractTensor, torch.Tensor)):
            return a.shape
        return ()  # Assume scalar e.g. 2.2 * torch.rand(2,3)

    shapes = [get_shape(a) for a in args]
    sh = torch.broadcast_shapes(*shapes)

    # Type
    def get_dtype(a):
        if isinstance(a, (torch.Tensor, AbstractTensor)):
            return a.dtype
        if isinstance(a, AbstractValue):
            return a.ty
        return torch.tensor(a).dtype

    dty = get_dtype(args[0])
    for arg in args:
        dty = torch.promote_types(dty, get_dtype(arg))

    return AbstractTensor(sh, dty)


def shnty_propagate_broadcast(op):
    """
    Make a propagator for OP, with normal Pytorch broadcasting semantics
    """
    return lambda *args: shnty_propagate_broadcast_aux(op, *args)


for op in (
    operator.mul,
    operator.truediv,
    operator.add,
    operator.neg,
    torch.relu,
    torch.atan2,
    torch.sin,
    torch.cos,
):
    shnty_propagator_register(op, shnty_propagate_broadcast(op))

for method in ("neg", "sin", "cos"):
    shnty_propagator_register((torch.Tensor, method), shnty_propagate_broadcast(op))

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


@shnty_propagator((torch.Tensor, "t"))
def _(x):
    if len(x.shape) != 2:
        raise NotImplementedError(f"check transpose implementation for snhty {x}")

    assert fx_is_tensor(x)
    sh = torch.Size((*x.shape[:-2], x.shape[-1], x.shape[-2]))
    return AbstractTensor(sh, x.dtype)


# --------------  @ matmul


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
        sh = ()  # sum over all elements, return type is scalar
        return AbstractValue(x.dtype)

    sh = torch.Size(s for i, s in enumerate(x.shape) if i not in dim)
    return AbstractTensor(sh, x.dtype)
