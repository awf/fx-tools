import pytest
import torch
from awfutils.pytree_utils import PyTree
import difffx as dfx
import vjp_rules
from fx_shnty import shnty_trace
from fx_print import fx_print

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
    t = t * torch.sum(t)
    t = t / torch.sum(t)
    tt = t.T
    tTt = tt @ t
    x = x @ tTt
    return torch.atan2(t, x)


@pytest.mark.parametrize("func,size", [(foo, (3, 3)), (my_func, (3, 5))])
def test_difffx(func, size):

    torch.manual_seed(42)

    x = torch.randn(*size)
    func(x)  # crash test
    fx_print(shnty_trace(func, (dfx.abstractify(x),)))

    foo_vjp = dfx.vjp(func, (dfx.abstractify(x),))
    fx_print(foo_vjp)

    dret = torch.randn_like(func(x))
    foo_vjp_pt = lambda x, dret: torch.autograd.functional.vjp(func, x, dret)

    PyTree.assert_close(foo_vjp_pt(x, dret), foo_vjp(x, dret))
    print("VJPs match OK")


if __name__ == "__main__":
    test_difffx(foo, (3, 3))
