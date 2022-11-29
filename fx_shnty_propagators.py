import operator
import torch

from fx_shnty import ShapeAndType, shnty_propagator, shnty_propagator_add

# --------------  Declare lots of propagators


def broadcast_shapes(arg1, *args):
    if len(args) == 0:
        return arg1
    else:
        return tuple(s for s in torch.broadcast_shapes(arg1, *args))


def shnty_propagate_broadcast_aux(op, arg0, *args):
    msg = f"shnty_propagate_broadcast_aux {op}"
    assert isinstance(arg0, ShapeAndType), msg
    assert all(isinstance(arg, ShapeAndType) for arg in args), msg
    sh = broadcast_shapes(arg0.sh, *(arg.sh for arg in args))
    dty = arg0.dtype_or_default_type()
    for arg in args:
        dty = torch.promote_types(dty, arg.dtype_or_default_type())

    if sh == ():
        return ShapeAndType(dty, (), None)
    else:
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
def _(x):
    assert x.isTensor
    return ShapeAndType(x.ty, (), x.dty)
