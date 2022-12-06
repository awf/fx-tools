import operator
import torch

from icecream import ic

from fx_shnty import (
    abstractify,
    get_return_abstract_value,
    shnty_trace,
    _shnty_propagator_dict,
)

from fx_print import fx_print


def test_shnty_0():
    def test(op, *args):
        val = op(*(v for v, _ in args))
        shntys = [vors for _, vors in args]
        sh = _shnty_propagator_dict[op](*shntys)
        assert abstractify(val) == sh

    def v(x):
        return x, x

    def s(x):
        return x, abstractify(x)

    for f1 in (s, v):
        for f2 in (s, v):
            test(operator.mul, f1(torch.randn(3, 4)), f2(torch.randn(3, 4)))
            test(operator.mul, f1(torch.randn(3, 4)), f2(5))
            test(operator.mul, f1(torch.randn(3, 4)), f2(5.5))


def test_shnty_1():
    def aux(p, q):
        return torch.relu(1.234 * p * q).neg()

    def my_func(x, b):
        y = 2 * x
        for _ in range(2):  # Loops will be unrolled
            x = aux(x, y)  # Function calls will be inlined
        t = b * y
        t = t * torch.sum(t)
        t = t / t.sum()
        tt = t.T
        tTt = tt @ t
        x = x @ tTt
        return torch.atan2(t, x)

    x = torch.rand(3, 5)
    x_shnty = abstractify(x)
    b = 3
    b_shnty = abstractify(b)

    ret = my_func(x, b)

    gm = shnty_trace(my_func, aargs=(x_shnty, b_shnty))

    print(gm.graph)
    fx_print(gm)
    print(get_return_abstract_value(gm))

    # Test return type
    assert get_return_abstract_value(gm) == abstractify(ret)


def test_shnty_2():
    # Test passing shapes
    def foo_a(x):
        return x * 14, x.shape

    def foo_b(y, tup):
        return y * tup[0]

    def foo(x):
        y, sh = foo_a(x)
        return foo_b(y, sh)

    x = torch.rand(3, 5)
    b = 3
    ret = foo(x)

    gm = shnty_trace(foo, (abstractify(x),))
    fx_print(gm)
    assert get_return_abstract_value(gm) == abstractify(ret)

    # Test tensor constants
    def foo_c(x):
        I = torch.eye(x.shape[1])
        return x @ I @ x.T

    ret = foo_c(x)

    gm = shnty_trace(foo_c, aargs=(abstractify(x),))
    fx_print(gm)
    assert get_return_abstract_value(gm) == abstractify(ret)

    # TODO test for notimplemented exceptions on function, method, module

    # TODO Test for poorly implemented propagator, e.g. returning 'int' rather than 'AbstractValue((), int)`


def test_shnty_3():

    from fx_shnty import shnty_trace, abstractify
    from fx_print import fx_print

    def aux(p, q):
        return torch.relu(1.234 * p * q).neg()

    def foo(x, b, n):
        y = b * x
        for _ in range(n):  # Loops will be unrolled
            x = aux(x, y)  # Function calls will be inlined
        return torch.atan2(y, x)

    x = torch.randn(3, 5)
    b = 8.2
    n = 2
    foo(x, b, n)

    foo_gm = shnty_trace(foo, (abstractify(x), abstractify(b), n))

    fx_print(foo_gm)


if __name__ == "__main__":
    test_shnty_3()
