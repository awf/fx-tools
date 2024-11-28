import pytest
import torch
from awfutils.pytree_utils import PyTree, pt_map
from awfutils import ndarray_str, pt_print
from fxtools import fx_print, fx_vjp


# Functions to vjp
def foo(x):
    w = torch.trace(x)
    # w = torch.sin(w)
    a = w * x + x
    return a


def aux(p, q):
    return torch.relu(1.234 * p * q).neg()


def my_func(x):
    y = 2 * x
    b = 4.4
    for _ in range(2):  # Loops will be unrolled
        x = aux(x, y)  # Function calls will be inlined
    t = b * y
    t = t * torch.sum(t) + x
    t = t / torch.sum(t)
    tTt = t.T @ t
    x = x @ tTt
    return torch.atan2(t, x)


@pytest.mark.parametrize("func,size", [(foo, (3, 3)), (my_func, (3, 5))])
def test_difffx(func, size):

    torch.manual_seed(42)

    x = torch.randn(*size)
    func(x)  # crash test

    dret = torch.randn_like(func(x))
    foo_vjp_pt = lambda x, dret: torch.autograd.functional.vjp(func, x, dret)[1]

    foo_vjp = fx_vjp(func, x)
    fx_print(foo_vjp)

    pt_print("d", foo_vjp(x, dret))

    PyTree.assert_close(foo_vjp_pt(x, dret), foo_vjp(x, dret))
    print("VJPs match OK")


@pytest.mark.skip("wip")
def test_multi():
    def foo(x, y):
        w = torch.trace(x) * y
        # w = torch.sin(w)
        a = w * x + x + y
        return a

    torch.manual_seed(42)

    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    ret = foo(x, y)

    dret = torch.randn_like(ret)
    dxy_pt = torch.autograd.functional.vjp(foo, (x, y), dret)[1]
    print()
    pt_print("vjp_pt", dxy_pt)

    foo_vjp = fx_vjp(foo, sample_input=(x, y))
    fx_print(foo_vjp)

    dxy = foo_vjp(x, y, dret)
    pt_print("vjp", dxy)

    PyTree.assert_close(dxy_pt, dxy)
    print("VJPs match OK")


if __name__ == "__main__":
    test_difffx(foo, (3, 3))
