import torch

from icecream import ic

from fx_shnty import fx_get_shnty, get_return_shnty, shnty_trace
import fx_shnty_propagators

from fx_print import fx_print


def test_shnty():
    def aux(p, q):
        return torch.relu(1.234 * p * q).neg()

    def my_func(x, b):
        y = 2 * x
        for _ in range(2):  # Loops will be unrolled
            x = aux(x, y)  # Function calls will be inlined
        t = b * y
        t = t * torch.sum(t)
        t = t / t.sum()
        x = x @ (t.T @ t)
        return torch.atan2(t, x)

    x = torch.rand(3, 5)
    x_shnty = fx_get_shnty(x)
    b = 3
    b_shnty = fx_get_shnty(b)
    print("xb", x_shnty, b_shnty)

    ret = my_func(x, b)

    gm = shnty_trace(my_func, arg_shntys=(x_shnty, b_shnty))

    print(gm.graph)
    fx_print(gm)
    print(get_return_shnty(gm))

    # Test return type
    assert get_return_shnty(gm) == fx_get_shnty(ret)

    # Test passing shapes
    def foo_a(x):
        return x * 14, x.shape

    def foo_b(y, tup):
        return y * tup[0]

    def foo(x):
        y, sh = foo_a(x)
        return foo_b(y, sh)

    ret == foo(x)

    gm = shnty_trace(foo, arg_shntys=(x_shnty,))
    fx_print(gm)
    assert get_return_shnty(gm) == fx_get_shnty(ret)

    # Test tensor constants
    def foo_c(x):
        I = torch.eye(x.shape[1])
        return x @ I @ x.T

    ret = foo_c(x)
    ic(ret)

    gm = shnty_trace(foo_c, arg_shntys=(x_shnty,))
    fx_print(gm)
    assert get_return_shnty(gm) == fx_get_shnty(ret)

    # TODO test for notimplemented exceptions on function, method, module

    # Test for poorly implemented propagator, e.g. returning 'int' rather than 'ShapeAndType((), int)`


if __name__ == "__main__":
    test_shnty()
