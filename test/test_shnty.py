import pytest

import operator
import torch

from fx_print import fx_print

from fx_shnty import (
    abstractify,
    fx_shape,
    fx_type,
    get_return_abstract_value,
    shnty_trace,
    _shnty_propagator_dict,
)


def _args(*args, **kwargs):
    return (args, kwargs)


def _aa(op, *args, **kwargs):
    return (op, (args, kwargs))


def _mm(*args, **kwargs):
    return (type(args[0]), (args, kwargs))


@pytest.mark.parametrize(
    "op_id,all_args",
    (
        _aa(torch.sum, torch.randn(3, 4)),
        _aa(torch.sum, torch.randn(3, 4), dim=(1,)),
        _aa(torch.sum, torch.randn(3, 4), dim=(1, 0)),
        _aa("sum", torch.randn(3, 4), dim=(1, 0)),
        _aa("cumsum", torch.randn(3, 4), dim=1),
        _aa("long", torch.randn(3, 4)),
        _aa("eq", torch.randn(3, 4), torch.randn(3, 4)),
        _aa(operator.mul, torch.randn(3, 4), torch.randn(3, 4)),
        _aa(operator.mul, torch.randn(3, 4), torch.tensor(5)),
        _aa(operator.mul, torch.randn(3, 4), torch.tensor(5.5)),
        _aa(operator.matmul, torch.randn(3, 4), torch.randn(4, 5)),
        _mm(
            torch.nn.Embedding(32768, 1023, 1),
            torch.randint(0, 32768, (3, 5)),
        ),
        _mm(torch.nn.LayerNorm(7), torch.randn(3, 5, 7)),
        _mm(
            torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(d_model=5 * 11, nhead=11), num_layers=7
            ),
            torch.randn(13, 17, 5 * 11),
        ),
    ),
)
def test_op(op_id, all_args):
    args = all_args[0]
    kwargs = all_args[1]

    if isinstance(op_id, str):
        # Method call
        ty = fx_type(args[0])
        op = getattr(args[0], op_id)
        val = op(*args[1:], **kwargs)
        shntys = map(abstractify, args)
        key = (ty, op_id)
        sh = _shnty_propagator_dict[key](*shntys, **kwargs)

    elif isinstance(op_id, type) and issubclass(op_id, torch.nn.Module):
        # Modyule call
        # Arg0 (the nn.Module) will always be a concrete value
        assert isinstance(args[0], op_id)
        key = op_id
        op = getattr(args[0], "forward")
        val = op(*args[1:], **kwargs)
        shntys = map(abstractify, args[1:])
        sh = _shnty_propagator_dict[key](args[0], *shntys, **kwargs)

    else:
        # Function call
        op = op_id
        val = op(*args, **kwargs)
        shntys = map(abstractify, args)
        sh = _shnty_propagator_dict[op](*shntys, **kwargs)

    assert abstractify(val) == sh


def test_shnty_abvals_vs_vals():
    def fn_test(op, *args, **kwargs):
        val = op(*(v for v, _ in args), **kwargs)
        shntys = [vors for _, vors in args]
        sh = _shnty_propagator_dict[op](*shntys, **kwargs)
        assert abstractify(val) == sh

    def meth_test(method, arg0, *args, **kwargs):
        op = getattr(arg0[0], method)
        val = op(*(v[0] for v in args), **kwargs)
        shntys = [vors[1] for vors in (arg0, *args)]
        key = (type(arg0[0]), method)
        sh = _shnty_propagator_dict[key](*shntys, **kwargs)
        assert abstractify(val) == sh

    def v(x):
        return x, x

    def s(x):
        return x, abstractify(x)

    for f1 in (s, v):
        fn_test(torch.sum, f1(torch.randn(3, 4)))
        fn_test(torch.sum, f1(torch.randn(3, 4)), dim=(1,))
        fn_test(torch.sum, f1(torch.randn(3, 4)), dim=(1, 0))
        meth_test("sum", f1(torch.randn(3, 4)))
        meth_test("sum", f1(torch.randn(3, 4)), dim=(1,))
        meth_test("reshape", f1(torch.randn(3, 4)), v([2, 6]))

    for f1 in (s, v):
        for f2 in (s, v):
            fn_test(operator.mul, f1(torch.randn(3, 4)), f2(torch.randn(3, 4)))
            fn_test(operator.mul, f1(torch.randn(3, 4)), f2(5))
            fn_test(operator.mul, f1(torch.randn(3, 4)), f2(5.5))

            fn_test(operator.matmul, f1(torch.randn(3, 5)), f2(torch.randn(5, 2)))


def test_shnty_bigfun():
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


def test_shnty_shape_passing():
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
        I = torch.eye(fx_shape(x)[1])
        return x @ I @ x.T

    ret = foo_c(x)

    gm = shnty_trace(foo_c, aargs=(abstractify(x),))
    fx_print(gm)
    assert get_return_abstract_value(gm) == abstractify(ret)

    # TODO test for notimplemented exceptions on function, method, module

    # TODO Test for poorly implemented propagator, e.g. returning 'int' rather than 'AbstractValue((), int)`


def test_shnty_concrete_vs_abstract():

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
    test_shnty_concrete_vs_abstract()
