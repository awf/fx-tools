from typing import Any
from collections import defaultdict
import torch

from icecream import ic

import torch.fx as tfx

from fx_print import fx_print
from fx_shnty import shnty_trace, get_return_shnty, fx_get_shnty
import fx_shnty_propagators


def ensure_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x,)


def shnty(x):
    """
    Get shape and type of argument x
    """
    return fx_get_shnty(x)


# ----------------


# A mapping from python function to (forward, backward)
ad_map = {}


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
            assert kwargs == None or len(kwargs) == 0

            if target not in ad_map:
                raise NotImplementedError(f"Need VJP rule for {target}")
            # Look up forward/backward functions in `ad_map`
            fwd, bwd = ad_map[target]
            # Call the fwd function, getting proxies for returns
            val, aux = fwd(*args)
            # In the backward pass, we will compute:
            #  d[args[0]],...,d[args[-1]] = bwd(aux, d{val})
            # So remember: (args, bwd, aux, val)
            # Note that all of these are symbolic, so it's cheap to remember them
            self.stack.append((args, bwd, aux, val))
            # And return the return value (an FX Proxy)
            return val

        def call_method(self, target, args, kwargs):
            raise NotImplementedError  # use method_to_function

        def get_attr(self, target, args, kwargs):
            raise NotImplementedError  # TODO

    # Grab the FX graph
    f_trace = shnty_trace(f, sample_input)

    fx_print(f_trace)
    ret_shnty = get_return_shnty(f_trace)

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
        for (args, bwd, aux, val) in reversed(ad.stack):
            dargs = bwd(aux, d[val])
            for (a, da) in zip(args, ensure_tuple(dargs)):
                d[a] += da
        # And return ret and J'*dret
        return ret, d[x]

    return shnty_trace(vjp_template, sample_input + (ret_shnty,))


import vjp_rules


def vjp_linear(f):
    """
    Construct fwd and bwd for f a linear function of x
    """

    def fwd(*args):
        return f(*args), None

    def bwd(_, dret):
        return f(dret)

    return fwd, bwd


import operator

# TODO: a decorator like shnty_propagate?
ad_map[operator.neg] = vjp_linear(operator.neg)
ad_map[operator.add] = (vjp_rules.add_fwd, vjp_rules.add_bwd)
ad_map[operator.mul] = (vjp_rules.mul_fwd, vjp_rules.mul_bwd)
ad_map[operator.matmul] = (vjp_rules.matmul_fwd, vjp_rules.matmul_bwd)
ad_map[torch.neg] = vjp_linear(torch.neg)
ad_map[torch.sin] = (vjp_rules.sin_fwd, vjp_rules.sin_bwd)
ad_map[torch.relu] = (vjp_rules.relu_fwd, vjp_rules.relu_bwd)
ad_map[torch.transpose] = (vjp_rules.transpose_fwd, vjp_rules.transpose_bwd)
ad_map[torch.diag] = (vjp_rules.diag_fwd, vjp_rules.diag_bwd)
ad_map[vjp_rules.scale] = (vjp_rules.scale_fwd, vjp_rules.scale_bwd)
ad_map[torch.trace] = (vjp_rules.trace_fwd, vjp_rules.trace_bwd)