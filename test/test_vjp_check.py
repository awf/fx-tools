import pytest
import torch
from fxtools import vjp_check_fwdbwd, vjp_check_fd


def test_vjp_check():
    def foo(a, T):
        return a * T, torch.sin(a)

    def foo_vjp_fwd(a, T):
        return foo(a, T), (a, T)

    def foo_vjp_bwd(aux, dret):
        a, T = aux
        da = torch.sum(T * dret[0])
        dT = a * dret[0]
        da += torch.cos(a) * dret[1]
        return da, dT

    vjp_check_fwdbwd(
        foo, foo_vjp_fwd, foo_vjp_bwd, (torch.tensor(1.123), torch.randn(3, 4))
    )


# Test that the check fails for a bad vjp
def test_vjp_check_bad():
    def foo(a, T):
        return a * T, torch.sin(a)

    def foo_vjp_fwd(a, T):
        return foo(a, T), (a, T)

    def foo_vjp_bwd_bad(aux, dret):
        a, T = aux
        da = torch.sum(T * dret[0])
        dT = a * dret[0]
        da += torch.sin(a) * dret[1]  # sin, not cos
        return da, dT

    with pytest.raises(AssertionError):
        vjp_check_fwdbwd(
            foo, foo_vjp_fwd, foo_vjp_bwd_bad, (torch.tensor(1.123), torch.randn(3, 4))
        )


def test_fd():
    # Function to vjp
    def foo(x):
        h = torch.trace(x)
        a = torch.sin(h) + x
        return a

    foo_vjp = lambda x, dret: torch.autograd.functional.vjp(foo, x, dret)

    vjp_check_fd(foo, foo_vjp, torch.randn(3, 3))
