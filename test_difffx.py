import torch
from awfutils.pytree_utils import PyTree
import difffx as dfx
import vjp_rules

def test_foo():
    # Function to vjp
    def foo(x):
        w = torch.trace(x)
        w = torch.sin(w)
        a = vjp_rules.scale(w, x)
        return a

    torch.manual_seed(42)

    x = torch.randn(3,3)
    foo_vjp = dfx.vjp(foo, x)

    dret = torch.randn_like(foo(x))
    foo_vjp_pt = lambda x,dret: torch.autograd.functional.vjp(foo, x, dret)

    PyTree.assert_close(foo_vjp_pt(x,dret), foo_vjp(x, dret))
    print('VJPs match OK')
