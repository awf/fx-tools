import operator
import torch
from vjp_check import vjp_check_fwdbwd
from fx_shnty import shnty_propagator, fx_shape
from icecream import ic

# Define a bunch of manual vjps
# Ultimately parse these out of https://github.com/pytorch/pytorch/blob/master/tools/autograd/derivatives.yaml

# SCALE
@torch.fx.wrap
def scale(a, T):
    return a * T


@shnty_propagator(scale)
def _(A, B):
    return B


def scale_fwd(a, T):
    return scale(a, T), (a, T)


def scale_bwd(aux, dret):
    # T: mxn
    # dret: mxn
    a, T = aux
    da = torch.sum(T * dret)
    dT = scale(a, dret)
    return da, dT


def test_scale():
    vjp_check_fwdbwd(
        scale, scale_fwd, scale_bwd, (torch.tensor(1.123), torch.randn(3, 4))
    )


# BS


def broadcast_shapes_with_expanders(*shapes):
    r"""broadcast_shapes_with_expanders(*shapes) -> (Size, Tuple[Size])

    A copy of torch.broadcast_shapes which returns expanders such that
        shapes[i].expand(expanders[i])
    represents the broadcast version of shapes[i]

    Args:
        \*shapes (torch.Size): Shapes of tensors.

    Returns:
        shape (torch.Size): A shape compatible with all input shapes.
        expanders (List[torch.Size]): Expanders as above.

    Raises:
        RuntimeError: If shapes are incompatible.
    """
    result = torch.broadcast_shapes(*shapes)

    def shape_to_expander(shape):
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

    expanders = [shape_to_expander(shape) for shape in shapes]

    return result, expanders


def test_broadcast_shapes():
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

            actual, expanders = broadcast_shapes_with_expanders(s0, s1)
            assertEqual(expected, actual)

            expected_val = torch.atan2(x0, x1)
            x0_expanded = x0.expand(expanders[0])
            x1_expanded = x1.expand(expanders[1])
            assert x0_expanded.shape == x1_expanded.shape
            actual_val = torch.atan2(x0_expanded, x1_expanded)
            assertEqual(expected_val, actual_val)

    inputs_list = [[1, 4], [4, 1], [1, 1, 3]]
    for integral_inputs in inputs_list:
        res1, _expanders = broadcast_shapes_with_expanders(*integral_inputs)
        res2 = torch.broadcast_tensors(*map(torch.empty, integral_inputs))[0].shape
        assertEqual(res1, res2)

    diff_input_types = [(1, (5,)), (3, (1,)), (1, (3, 4))]
    for s0 in diff_input_types:
        res1, _expanders = broadcast_shapes_with_expanders(*s0)
        res2 = torch.broadcast_tensors(*map(torch.empty, s0))[0].shape
        assertEqual(res1, res2)


# MUL


def mul_fwd(A, B):
    return A * B, (A, B)


def mul_bwd(aux, dret):
    A, B = aux
    shape, (expanderA, expanderB) = broadcast_shapes_with_expanders(
        fx_shape(A), fx_shape(B)
    )

    def contract_over_expanded_dims(X, expander):
        dims_to_sum = tuple(i for i, v in enumerate(expander) if v != -1)
        if len(dims_to_sum):
            return X.sum(dims_to_sum)
        else:
            return X

    dA = contract_over_expanded_dims(B * dret, expanderA)
    dB = contract_over_expanded_dims(A * dret, expanderB)
    return (dA, dB)


def test_mul():
    vjp_check_fwdbwd(
        operator.mul, mul_fwd, mul_bwd, (torch.randn(3, 4), torch.randn(3, 4))
    )
    vjp_check_fwdbwd(
        operator.mul, mul_fwd, mul_bwd, (torch.tensor(3.14159), torch.randn(3, 4))
    )
    # works for difffx, not for torch:    vjp_check_fwdbwd(operator.mul, mul_fwd, mul_bwd, (3.14159, torch.randn(3, 4)))


def matmul_fwd(A, B):
    return A @ B, (A, B)


def matmul_bwd(aux, dret):
    # A: mxp
    # B: pxn
    # dret: mxn
    A, B = aux
    dA = dret @ B.T
    dB = A.T @ dret
    return dA, dB


def test_matmul():
    vjp_check_fwdbwd(
        operator.matmul, matmul_fwd, matmul_bwd, (torch.randn(3, 4), torch.randn(4, 5))
    )


# Add
def add_fwd(A, B):
    return A + B, None


def add_bwd(aux, dret):
    return dret, dret


def test_add():
    vjp_check_fwdbwd(
        operator.add, add_fwd, add_bwd, (torch.randn(3, 4), torch.randn(3, 4))
    )


# Sin
def sin_fwd(x):
    return (torch.sin(x), x)


def sin_bwd(aux_is_x, dret):
    return torch.cos(aux_is_x) * dret


def test_sin():
    vjp_check_fwdbwd(torch.sin, sin_fwd, sin_bwd, (torch.randn(3, 4),))


# Relu forward pass - save sign(x), could save as uint8 if we wanted to save memory
def relu_fwd(x):
    return torch.relu(x), (x > 0)


def relu_bwd(aux, dret):
    return aux * dret


def test_relu():
    vjp_check_fwdbwd(torch.relu, relu_fwd, relu_bwd, (torch.randn(3, 4),))


# Neg
def neg_fwd(x):
    return -x, None


def neg_bwd(aux, dret):
    return -dret


def test_neg():
    vjp_check_fwdbwd(torch.neg, neg_fwd, neg_bwd, (torch.randn(3, 4),))


# Trace
def trace_fwd(x):
    return torch.trace(x), x.shape


def trace_bwd(x_shape, dret):
    return dret * torch.eye(*x_shape)


def test_trace():
    vjp_check_fwdbwd(torch.trace, trace_fwd, trace_bwd, (torch.randn(3, 3),))


# Diag
def diag_fwd(x):
    return torch.diag(x), x.shape


def diag_bwd(shape, dret):
    return torch.diag(dret)


def test_diag():
    vjp_check_fwdbwd(torch.diag, diag_fwd, diag_bwd, (torch.randn(3, 3),))


# transpose
def transpose(x):
    return torch.transpose(x, 0, 1)


def transpose_fwd(x):
    return transpose(x), None


def transpose_bwd(aux, dret):
    return transpose(dret)


def test_transpose():
    vjp_check_fwdbwd(transpose, transpose_fwd, transpose_bwd, (torch.randn(3, 5),))


# def transpose_fwd(x,m,n):
#     return torch.transpose(x,m,n), (m,n)


# def transpose_bwd(aux, dret):
#     m,n = aux
#     return (torch.transpose(dret, m,n), None, None)
