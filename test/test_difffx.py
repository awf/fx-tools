import pytest
import torch
from awfutils.pytree_utils import PyTree, pt_map
from awfutils import ndarray_str
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
    # fx_print(shnty_trace(func, (dfx.abstractify(x),)))

    dret = torch.randn_like(func(x))
    foo_vjp_pt = lambda x, dret: torch.autograd.functional.vjp(func, x, dret)

    foo_vjp = fx_vjp(func, x)
    fx_print(foo_vjp)

    print(*pt_map(lambda x: ndarray_str(x.numpy()), foo_vjp(x, dret)), sep="\n")

    PyTree.assert_close(foo_vjp_pt(x, dret), foo_vjp(x, dret))
    print("VJPs match OK")


if __name__ == "__main__":
    test_difffx(foo, (3, 3))
