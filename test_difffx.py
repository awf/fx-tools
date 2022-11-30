import torch
from awfutils.pytree_utils import PyTree
import difffx as dfx
import vjp_rules
from fx_shnty import shnty_trace
from fx_print import fx_print


def test_difffx_1():
    # Function to vjp
    def foo(x):
        w = torch.trace(x)
        w = torch.sin(w)
        a = w * x
        return a

    torch.manual_seed(42)

    x = torch.randn(3, 3)
    foo(x)  # crash test
    fx_print(shnty_trace(foo, (dfx.shnty(x),)))

    foo_vjp = dfx.vjp(foo, (dfx.shnty(x),))

    dret = torch.randn_like(foo(x))
    foo_vjp_pt = lambda x, dret: torch.autograd.functional.vjp(foo, x, dret)

    PyTree.assert_close(foo_vjp_pt(x, dret), foo_vjp(x, dret))
    print("VJPs match OK")


if __name__ == "__main__":
    test_difffx_1()
