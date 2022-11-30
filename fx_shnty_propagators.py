import operator
import torch

from fx_shnty import ShapeAndType, shnty_propagator, shnty_propagator_add

# --------------  Declare lots of propagators


def is_shnty(x):
    return isinstance(x, ShapeAndType)


def broadcast_shapes(arg1, *args):
    if len(args) == 0:
        return arg1
    else:
        sh = torch.broadcast_shapes(arg1, *args)
        return tuple(s for s in sh)


def shnty_propagate_broadcast_aux(op, *args):
    msg = f"shnty_propagate_broadcast_aux {op}"

    # Shape
    shapes = [a.sh if is_shnty(a) else () for a in args]
    sh = broadcast_shapes(*shapes)

    # Type
    def get_type(a):
        return a.dtype_or_default_type() if is_shnty(a) else type(a)

    dty = get_type(args[0])
    for arg in args:
        dty = torch.promote_types(dty, get_type(arg))

    return ShapeAndType(torch.Tensor, sh, dty)


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
    shnty_propagator_add(op, shnty_propagate_broadcast(op))

for method in ("neg", "sin", "cos"):
    shnty_propagator_add((torch.Tensor, method), shnty_propagate_broadcast(op))


@shnty_propagator(torch.trace)
def _(x):
    if len(x.sh) != 2:
        raise NotImplementedError(
            f"check trace implementation for snhty {x}, it might be fine"
        )
    assert x.isTensor
    return ShapeAndType(x.ty, x.sh[:-2], x.dty)


@shnty_propagator((torch.Tensor, "t"))
def _(x):
    if len(x.sh) != 2:
        raise NotImplementedError(f"check transpose implementation for snhty {x}")

    assert x.isTensor
    sh = (*x.sh[:-2], x.sh[-1], x.sh[-2])
    return ShapeAndType(x.ty, sh, x.dty)


@shnty_propagator(operator.matmul)
def _(A, B):
    assert A.isTensor and B.isTensor
    # m x p times p x n
    m, pA = A.sh[-2:]
    pB, n = B.sh[-2:]
    leading_dims = A.sh[:-2]
    assert pA == pB
    assert leading_dims == B.sh[:-2]
    assert A.dty == B.dty  # TODO: promotions
    return ShapeAndType(A.ty, (*leading_dims, m, n), A.dty)


@shnty_propagator((torch.Tensor, "sum"))
@shnty_propagator(torch.sum)
def _(x, dim=None):
    assert x.isTensor
    if not dim or len(dim) == 0:
        sh = ()  # sum over all elements
    else:
        sh = tuple(s for i, s in enumerate(x.sh) if i not in dim)
    return ShapeAndType(x.ty, sh, x.dty)
