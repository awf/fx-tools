from types import GetSetDescriptorType
from collections import defaultdict
import torch

from icecream import ic

import torch.fx as tfx

from fx_print import fx_print, fn_name
from fx_shnty import (
    abstractify,  # for re-export
    AbstractValue,
    shnty_trace,
    get_return_abstract_value,
    fx_type,
    fx_shape,
    _opspec_to_str,
)


def ensure_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x,)


# ----------------


# A mapping from python function to (forward, backward)
_ad_map = {}


def vjp(f, sample_input):
    """
    An FX transform that implements reverse-mode automatic differentiation.

    >>>


    If the traced function is of the form
    ```py
    def foo(a1..an):
      t1 = f1(a1..an)
      t2 =  f2(a1..an,t1) # wlog, fk uses all in-scope variables
      ...
      tm =   fm(a1..an,t1..t{m-1})
      return tm
    ```
    Then the VJP (vector-jacobian product is of the form)
    ```py
    def foo_vjp(a1..an, dtm):
      t1, aux1 = f1_fwd(a1..an)
      t2, aux2 = f2_fwd(a1..an,t1)
      ...
      tm, auxm = fm_fwd(a1..an,t1..{m-1})

      da{1..n},dt{1..m-1} += fm_bwd(auxm, dtm)
      ...
      da{1..n},dt1 += f2_bwd(aux2, dt3)
      da{1..n} += f1_bwd(aux1, dt1)

      return da{1..n}
    ```
    """

    # Check that sample inputs are tuples, and that at least one is an AbstractValue
    if not isinstance(sample_input, tuple):
        raise ValueError("sample input should be a tuple of [abstract] values")
    if not any([isinstance(a, AbstractValue) for a in sample_input]):
        raise ValueError("sample input should contain at least one abstract value")

    # if len(sample_input) > 1:
    #     raise ValueError("Annoyingly, vjp doesn't work for multiple inputs")
    #     # Problem is that the interpreter below can't sensibly deal with *args
    #     # A Solution: rewrite the top level of `trace` (called by shnty_trace)
    #     # to not use co_argcount, but to use the length of the supplied args.

    class ADInterpreter(torch.fx.Interpreter):
        """
        This interpreter runs through the forward transformation,
        replacing calls to `fk` with `fk_fwd = ad_map[fk][0]`,
        and recording the operations on a stack.
        """

        def __init__(self, f):
            super().__init__(f)
            self.stack = []

        def call_function(self, target, args, kwargs):
            # Assume kwargs are not to be differentiated

            # translate getattrs
            if target == getattr:
                assert len(args) == 2
                attr = args[1]
                assert isinstance(attr, str)
                if attr == "shape":
                    raise NotImplementedError(
                        "To use diffx, consider using `fx_shape(x)` instead of `x.shape`"
                    )

                # get the class method.  If the class has overridden __getattr__,
                # this may fail.  Might need to whitelist types where this is OK,
                # e.g. torch.Tensor
                target = getattr(fx_type(args[0]), args[1])
                assert isinstance(target, GetSetDescriptorType)
                args = args[:1]  # drop attr

            # Look up forward/backward functions in `ad_map`
            if target not in _ad_map:
                msg = f"Need VJP rule for {_opspec_to_str(target)}"
                print(msg)
                raise NotImplementedError(msg)

            fwd, bwd = _ad_map[target]

            # Call the fwd function, getting proxies for returns
            val, aux = fwd(*args, **kwargs)

            # In the backward pass, we will compute:
            #  d[args[0]],...,d[args[-1]] = bwd(aux, d{val})
            # So remember: (args, bwd, aux, val)
            # Note that all of these are symbolic, so it's cheap to remember them
            self.stack.append((args, bwd, aux, val))
            # And return the return value (an FX Proxy)
            return val

        def call_method(self, target, args, kwargs):
            key = (fx_type(args[0]), target)
            return self.call_function(key, args, kwargs)

        # def get_attr(self, target, args, kwargs):
        #     raise NotImplementedError  # TODO

    # Grab the FX graph
    f_trace = shnty_trace(f, sample_input)

    fx_print(f_trace)

    # Get abstract return value
    ret_shnty = get_return_abstract_value(f_trace)

    # This is the "template" function for the VJP
    def vjp_template(x, dret):
        # Run the forward computations, and collect them in ad.stack
        ad = ADInterpreter(f_trace)
        ret = ad.run(x)
        # Build a dict to hold derivatives
        d = {}
        # Add dret to derivatives dict
        d[ret] = dret
        # And run down the stack...
        for (args, bwd, aux, val) in reversed(ad.stack):
            dval = d[val] if val in d else torch.zeros(fx_shape(val))
            dargs = bwd(aux, dval)
            for (a, da) in zip(args, ensure_tuple(dargs)):
                if fx_shape(a) != fx_shape(da):
                    raise ValueError(
                        f"Derivative has different shape from arg: "
                        + f"{fx_shape(da)} != {fx_shape(a)}, "
                        + f"in {fn_name(bwd)}"
                    )
                if a in d:
                    d[a] = d[a] + da
                else:
                    d[a] = da
        # And return ret and J'*dret
        return ret, d[x]

    vjp_args = sample_input + (ret_shnty,)

    return shnty_trace(vjp_template, vjp_args)


def register_vjp_rule(opspec, fwd, bwd):
    """
    Register fwd and bwd for function f or method (Type, "foo")
    """

    assert opspec not in _ad_map
    _ad_map[opspec] = (fwd, bwd)


def register_vjp_rule_linear(f):
    """
    Construct fwd and bwd for f a linear function of x
    """

    assert not isinstance(f, tuple)

    def fwd(*args):
        return f(*args), None

    def bwd(_, dret):
        return f(dret)

    register_vjp_rule(f, fwd, bwd)


def set_name(func, name):
    func.__name__ = name
    func.__qualname__ = func.__module__ + "." + name


def vjp_rule_fwd(opspec):
    """ """

    def the_decorator(func):
        if opspec in _ad_map:
            assert _ad_map[opspec][0] == None  # Fwd not yet registered
            _ad_map[opspec][0] = func
        else:
            _ad_map[opspec] = [func, None]

        # Override name, normally just '_'
        if func.__name__ == "_":
            set_name(func, f"vjp_fwd<{_opspec_to_str(opspec)}>")

        return func  # normally just gets assigned to '_'

    return the_decorator


def vjp_rule_bwd(opspec):
    """ """

    def the_decorator(func):
        if opspec in _ad_map:
            assert _ad_map[opspec][1] == None  # Bwd not yet registered
            _ad_map[opspec][1] = func
        else:
            _ad_map[opspec] = [None, func]

        # Override name, normally just '_'
        if func.__name__ == "_":
            set_name(func, f"vjp_bwd<{_opspec_to_str(opspec)}>")

        return func  # normally just gets assigned to '_'

    return the_decorator


import vjp_rules
