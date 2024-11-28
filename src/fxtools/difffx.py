from types import GetSetDescriptorType
from typing import Any, Callable, Type

from collections import defaultdict

import numpy as np
import torch

import torch.fx as tfx
import torch.fx.passes

from .fx_print import fx_print, fn_name


def ensure_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x,)


class AnnotatingInterpreter(torch.fx.Interpreter):
    """
    An FX Interpreter that attaches the original FX node to proxies.

    This allows annotations left by previous passes to be picked up, for example shapes
    """

    def run_node(self, n):
        val = super().run_node(n)
        val.node.meta["dfx_orig_node"] = n  # Attach node to val
        return val


def fx_add_shapes(f_trace: torch.fx.GraphModule, sample_input: Any):
    """
    Run shape propagation on graph `f_trace`, which will add shape metadata in place.
    """
    torch.fx.passes.graph_manipulation.ShapeProp(f_trace).run(*sample_input)


def fx_type(x):
    if isinstance(x, torch.fx.Proxy):
        if "dfx_orig_node" not in x.node.meta:
            raise ValueError(f"Node {x.node} has no shape metadata")
        onode = x.node.meta["dfx_orig_node"]
        return onode.meta["type"]

    raise ValueError(f"Unhandled  {x}")


def fx_shape(x):
    """
    Return the shape of tensor or FX Proxy x.  Using `fx_shape` instead of `x.shape`
    means that the shape can be extracted from Proxies as well as Tensors.

    Assumes that ShapeProp has been run on the graph, so that x.fsi_node is set
    """
    if isinstance(x, torch.fx.Proxy):
        if "dfx_orig_node" not in x.node.meta:
            raise ValueError(f"Node {x.node} has no shape metadata")
        onode = x.node.meta["dfx_orig_node"]
        return onode.meta["tensor_meta"].shape
    elif isinstance(x, torch.Tensor):
        return x.shape
    elif np.issubdtype(type(x), np.number):
        return ()  # Compatible with torch.broadcast_shapes
    else:
        raise ValueError(f"Unhandled type {type(x)}")


def _canonicalize_class(x):
    if x.__class__ == torch.Tensor.__class__:
        return torch.Tensor  # not _TensorBase
    else:
        return x.__class__


def _classname(x):
    return _canonicalize_class(x).__qualname__


def _opspec_to_str(opspec):
    if isinstance(opspec, Callable):
        return f"function `{fn_name(opspec)}`"

    if isinstance(opspec, tuple):
        ty, attr = opspec
        assert isinstance(ty, Type)
        assert isinstance(attr, str)
        return f"method `{fn_name(ty)}.{attr}`"

    if isinstance(opspec, GetSetDescriptorType):
        return f"attr `{_classname(opspec.__objclass__)}.{opspec.__name__}`"

    assert False


# ----------------


# A mapping from python function to (forward, backward)
_ad_map = {}


def vjp(f, sample_input):
    """
    An FX transform that implements reverse-mode automatic differentiation.

    The function `f` should take n tensor arguments, and return a single tensor
    or a tuple of tensors.

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

    class ADInterpreter(AnnotatingInterpreter):
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
            key = (fx_type(args[0]), target)  # assumes all nodes are torch.Tensor
            return self.call_function(key, args, kwargs)

        def get_attr(self, target, args, kwargs):
            raise NotImplementedError  # TODO

    # Grab the FX graph
    f_trace = torch.fx.symbolic_trace(f)

    # Run shape analysis, record answers in the graph
    torch.fx.passes.graph_manipulation.ShapeProp(f_trace).run(
        *ensure_tuple(sample_input)
    )

    # This is the "template" function for the VJP
    def vjp_template(x, dret):
        # Run the forward computations, and collect them in ad.stack
        ad = ADInterpreter(f_trace)
        ret = ad.run(x)
        # Build a dict to hold derivatives
        d = defaultdict(lambda: 0)
        # Add dret to derivatives dict
        d[ret] = dret
        # And run down the stack...
        for args, bwd, aux, val in reversed(ad.stack):
            dargs = bwd(aux, d[val])
            for a, da in zip(args, ensure_tuple(dargs)):
                d[a] += da
        # And return J'*dret
        return d[x]

    # Trace through vjp_template and return.
    return torch.fx.symbolic_trace(vjp_template)


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
    """
    The decorated function is the forward pass of the VJP rule for `opspec`.
    """

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
    """
    The decorated function is the backward pass of the VJP rule for `opspec`.
    """

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
