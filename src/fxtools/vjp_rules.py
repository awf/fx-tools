import operator
import torch
from vjp_check import vjp_check_fwdbwd

from difffx import (
    vjp_rule_fwd,
    vjp_rule_bwd,
    register_vjp_rule_linear,
    register_vjp_rule,
    fx_shape,
    _ad_map,
)

# Define a bunch of manual vjps
# Ultimately parse these out of https://github.com/pytorch/pytorch/blob/master/tools/autograd/derivatives.yaml

# Reminder: for
#  def foo(x : S) -> T
# we define
#  def foo_fwd(x : S) -> (T, Aux_foo)
#  def foo_bwd(aux : Aux_foo, dret : dT) -> dS


def check_op(op, *args):
    fwd, bwd = _ad_map[op]
    if isinstance(op, tuple):
        op = getattr(op[0], op[1])  # Decode method
        vjp_check_fwdbwd(op, fwd, bwd, args)
    else:
        vjp_check_fwdbwd(op, fwd, bwd, args)


# SCALE
@torch.fx.wrap
def scale(a, T):
    return a * T


@vjp_rule_fwd(scale)
def _(a, T):
    return scale(a, T), (a, T)


@vjp_rule_bwd(scale)
def _(aux, dret):
    # T: mxn
    # dret: mxn
    a, T = aux
    da = torch.sum(T * dret)
    dT = scale(a, dret)
    return da, dT


def test_scale():
    check_op(scale, 1.3, torch.randn(3, 4))


# BS


def shape_to_expander(result, shape):
    """shape_to_expander(result, shape) -> torch.Size
    Given that argument `shape` has been broadcast to `result`,
    return `dims` such that
      t.expand(dims)
    has shape `result` where `t` is a tensor of shape `shape`.
    """
    if isinstance(shape, int):
        shape = (shape,)
    expander = list(r for r in result)
    for i in range(-1, -1 - len(shape), -1):
        if shape[i] == 1:
            expander[i] = result[i]
        else:
            assert shape[i] == result[i]
            expander[i] = -1
    return expander


def test_shape_to_expander():
    def assertEqual(A, B):
        if isinstance(A, torch.Tensor):
            assert torch.equal(A, B)
        else:
            assert A == B

    examples = [(), (1,), (2,), (1, 1), (3, 1), (3, 2), (4, 1, 1), (4, 3, 2)]
    for s0 in examples:
        for s1 in examples:
            x0 = torch.randn(s0)
            x1 = torch.randn(s1)
            expected = torch.broadcast_tensors(x0, x1)[0].shape

            actual = torch.broadcast_shapes(s0, s1)
            assertEqual(expected, actual)

            expected_val = torch.atan2(x0, x1)
            x0_expanded = x0.expand(shape_to_expander(actual, s0))
            x1_expanded = x1.expand(shape_to_expander(actual, s1))
            assert x0_expanded.shape == x1_expanded.shape
            actual_val = torch.atan2(x0_expanded, x1_expanded)
            assertEqual(expected_val, actual_val)

    inputs_list = [[1, 4], [4, 1], [1, 1, 3]]
    for integral_inputs in inputs_list:
        res1 = torch.broadcast_shapes(*integral_inputs)
        res2 = torch.broadcast_tensors(*map(torch.empty, integral_inputs))[0].shape
        assertEqual(res1, res2)

    diff_input_types = [(1, (5,)), (3, (1,)), (1, (3, 4))]
    for s0 in diff_input_types:
        res1 = torch.broadcast_shapes(*s0)
        res2 = torch.broadcast_tensors(*map(torch.empty, s0))[0].shape
        assertEqual(res1, res2)


# How to contract derivatives for broadcasting ops
# TODO: FX transform to remove implicit broadcasting
def contract_over_expanded_dims(X, expander):
    dims_to_sum = tuple(i for i, v in enumerate(expander) if v != -1)
    if len(dims_to_sum):
        return X.sum(dims_to_sum)
    else:
        return X


def contract_after_broadcast(A, B, dA, dB):
    A_shape = fx_shape(A)
    B_shape = fx_shape(B)
    shape = torch.broadcast_shapes(A_shape, B_shape)

    dAcontracted = contract_over_expanded_dims(dA, shape_to_expander(shape, A_shape))
    dBcontracted = contract_over_expanded_dims(dB, shape_to_expander(shape, B_shape))
    return (dAcontracted.reshape(A_shape), dBcontracted.reshape(B_shape))


# Add
@vjp_rule_fwd(operator.add)
def _(A, B):
    ret = A + B
    # inlining contract_after_broadcast so we don't stash A and B
    A_shape = fx_shape(A)
    B_shape = fx_shape(B)
    shape = torch.broadcast_shapes(A_shape, B_shape)
    assert fx_shape(ret) == shape if isinstance(ret, torch.Tensor) else True

    return ret, (A_shape, B_shape, shape)


@vjp_rule_bwd(operator.add)
def _(aux, dret):
    A_shape, B_shape, shape = aux
    expanderA = shape_to_expander(shape, A_shape)
    expanderB = shape_to_expander(shape, B_shape)
    dA = contract_over_expanded_dims(dret, expanderA)
    dB = contract_over_expanded_dims(dret, expanderB)
    # Reshapes are to canonicalize e.g. Size([]) vs Size([1])
    return (dA.reshape(A_shape), dB.reshape(B_shape))


def test_add():
    check_op(operator.add, *(torch.randn(3, 4), torch.randn(3, 4)))
    check_op(operator.add, *(torch.randn(1), torch.randn(3, 4)))
    check_op(operator.add, *(torch.randn(3, 1), torch.randn(3, 4)))


# MUL


@vjp_rule_fwd(operator.mul)
def _(A, B):
    return A * B, (A, B)


@vjp_rule_bwd(operator.mul)
def _(aux, dret):
    A, B = aux
    return contract_after_broadcast(A, B, B * dret, A * dret)


def test_mul():
    check_op(operator.mul, *(torch.randn(3, 4), torch.randn(3, 4)))
    check_op(operator.mul, *(torch.tensor(3.14159), torch.randn(3, 4)))
    # works for difffx, not for torch:    check_op(operator.mul, *(3.14159, torch.randn(3, 4)))


# Div
@vjp_rule_fwd(operator.truediv)
def _(A, B):
    return A / B, (A, B)


@vjp_rule_bwd(operator.truediv)
def _(aux, dret):
    A, B = aux
    return contract_after_broadcast(A, B, dret / B, -dret * A / (B * B))


def test_div():
    check_op(operator.truediv, *(torch.randn(3, 4), torch.rand(3, 4) + 0.1))
    check_op(operator.truediv, *(torch.randn(3, 1), torch.rand(3, 4) + 0.1))
    check_op(operator.truediv, *(torch.randn(3, 4), torch.rand(1, 4) + 0.1))
    check_op(operator.truediv, *(torch.randn(3, 4), 0.1))


# MatMul


@vjp_rule_fwd(operator.matmul)
def _(A, B):
    return A @ B, (A, B)


@vjp_rule_bwd(operator.matmul)
def _(aux, dret):
    # A: mxp
    # B: pxn
    # dret: mxn
    A, B = aux
    dA = dret @ B.T
    dB = A.T @ dret
    return dA, dB


def test_matmul():
    check_op(operator.matmul, *(torch.randn(3, 4), torch.randn(4, 5)))
    # TODO: check_op(operator.matmul, *(torch.randn(2, 3, 4), torch.randn(2, 4, 5)))


# Sum

# def _(x, dim=None):
#     assert fx_is_tensor(x)

#     if not dim or len(dim) == 0:
#         sh = ()  # sum over all elements, return type is scalar
#         return AbstractValue(x.dtype)

#     sh = torch.Size(s for i, s in enumerate(x.shape) if i not in dim)
#     return AbstractTensor(sh, x.dtype)


@vjp_rule_fwd(torch.sum)
@vjp_rule_fwd((torch.Tensor, "sum"))
def _(A, dim=None):
    return torch.sum(A, dim), (dim, fx_shape(A))


@vjp_rule_bwd(torch.sum)
@vjp_rule_bwd((torch.Tensor, "sum"))
def _(aux, dret):
    dim, shape = aux

    assert not dim  # Need more amusing logic to deal with dim

    return dret * torch.ones(shape)


def test_sum():
    check_op(torch.sum, *(torch.randn(3, 4),))


# Cos
@vjp_rule_fwd(torch.cos)
def _(x):
    return (torch.cos(x), x)


@vjp_rule_bwd(torch.cos)
def _(aux_is_x, dret):
    return -torch.sin(aux_is_x) * dret


def test_cos():
    check_op(torch.cos, *(torch.randn(3, 4),))


# Sin
@vjp_rule_fwd(torch.sin)
def _(x):
    return (torch.sin(x), x)


@vjp_rule_bwd(torch.sin)
def _(aux_is_x, dret):
    return torch.cos(aux_is_x) * dret


def test_sin():
    check_op(torch.sin, *(torch.randn(3, 4),))


if False:
    # Demonstration of how to implement an alternative definition of sin, optimized for time.
    sin_Ot = lambda x: torch.sin(x)

    @vjp_rule_fwd(sin_Ot)
    def _(x):
        s, c = torch.sincos(x)
        return (s, c)

    @vjp_rule_bwd(sin_Ot)
    def _(c, dret):
        return c * dret

    def test_sin_Ot():
        check_op(sin_Ot, *(torch.randn(3, 4),))


# Atan2
@vjp_rule_fwd(torch.atan2)
def _(x, y):
    return torch.atan2(x, y), (x, y)


@vjp_rule_bwd(torch.atan2)
def _(aux, dret):
    x, y = aux
    tmp = dret / (x * x + y * y)
    dx = y * tmp
    dy = -x * tmp
    return (dx, dy)


def test_atan2():
    check_op(torch.atan2, *(torch.randn(3, 4), torch.randn(3, 4)))


# Relu forward pass - save sign(x), could save as uint8 if we wanted to save memory
@vjp_rule_fwd(torch.relu)
def _(x):
    return torch.relu(x), (x > 0)


@vjp_rule_bwd(torch.relu)
def _(aux, dret):
    return aux * dret


def test_relu():
    check_op(torch.relu, *(torch.randn(3, 4),))


@vjp_rule_fwd((torch.Tensor, "neg"))
def _(x):
    return -x, None


@vjp_rule_bwd((torch.Tensor, "neg"))
def _(aux, dret):
    return -dret


# Neg
register_vjp_rule_linear(operator.neg)
register_vjp_rule_linear(torch.neg)


def test_neg():
    check_op(operator.neg, torch.randn(3, 4))
    check_op(torch.neg, torch.randn(3, 4))


# Trace
@vjp_rule_fwd(torch.trace)
def _(x):
    return torch.trace(x), fx_shape(x)


@vjp_rule_bwd(torch.trace)
def _(x_shape, dret):
    return dret * torch.eye(*x_shape)


def test_trace():
    check_op(torch.trace, *(torch.randn(3, 3),))


# Diag
@vjp_rule_fwd(torch.diag)
def _(x):
    return torch.diag(x), fx_shape(x)


@vjp_rule_bwd(torch.diag)
def _(shape, dret):
    return torch.diag(dret)


def test_diag():
    check_op(torch.diag, *(torch.randn(3, 3),))


# transpose
def transpose(x):
    return torch.transpose(x, 0, 1)


@vjp_rule_fwd(torch.Tensor.T)
@vjp_rule_fwd((torch.Tensor, "t"))
@vjp_rule_fwd(transpose)
def _(x):
    return transpose(x), None


@vjp_rule_bwd(torch.Tensor.T)
@vjp_rule_bwd((torch.Tensor, "t"))
@vjp_rule_bwd(transpose)
def _(aux, dret):
    return transpose(dret)


def test_transpose():
    check_op(transpose, *(torch.randn(3, 5),))


@vjp_rule_fwd(torch.transpose)
def _(x, m, n):
    return torch.transpose(x, m, n), (m, n)


@vjp_rule_bwd(torch.transpose)
def _(aux, dret):
    m, n = aux
    return (torch.transpose(dret, m, n), None, None)
