import operator
import torch
from vjp_check import vjp_check_fwdbwd

# Define a bunch of manual vjps
# Ultimately parse these out of https://github.com/pytorch/pytorch/blob/master/tools/autograd/derivatives.yaml


@torch.fx.wrap
def scale(a, T):
    return a * T


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


def mul_fwd(A, B):
    return A * B, (A, B)


def mul_bwd(aux, dret):
    A, B = aux
    return (B * dret, A * dret)


def test_mul():
    vjp_check_fwdbwd(
        operator.mul, mul_fwd, mul_bwd, (torch.randn(3, 4), torch.randn(3, 4))
    )


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


def add_fwd(A, B):
    return A + B, None


def add_bwd(aux, dret):
    return dret, dret


def test_add():
    vjp_check_fwdbwd(
        operator.add, add_fwd, add_bwd, (torch.randn(3, 4), torch.randn(3, 4))
    )


# Relu forward pass - save sign(x), could save as uint8 if we wanted to save memory
def relu_fwd(x):
    return torch.relu(x), (x > 0)


def relu_bwd(aux, dret):
    return aux * dret


def test_relu():
    vjp_check_fwdbwd(torch.relu, relu_fwd, relu_bwd, (torch.randn(3, 4),))


def neg_fwd(x):
    return -x, None


def neg_bwd(aux, dret):
    return -dret


def test_neg():
    vjp_check_fwdbwd(torch.neg, neg_fwd, neg_bwd, (torch.randn(3, 4),))


def trace_fwd(x):
    return torch.trace(x), x.shape


def trace_bwd(shape, dret):
    return dret * torch.eye(*shape)


def test_trace():
    vjp_check_fwdbwd(torch.trace, trace_fwd, trace_bwd, (torch.randn(3, 3),))


def diag_fwd(x):
    return torch.diag(x), x.shape


def diag_bwd(shape, dret):
    return torch.diag(dret)


def test_diag():
    vjp_check_fwdbwd(torch.diag, diag_fwd, diag_bwd, (torch.randn(3, 3),))
