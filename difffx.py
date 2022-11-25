from typing import Any
import torch.fx.passes

class AnnotatingInterpreter(torch.fx.Interpreter):
    """
    An FX Interpreter that attaches the original FX node to proxies.

    This allows annotations left by previous passes to be picked up, for example shapes
    """
    def run_node(self, n):
        val = super().run_node(n)
        val.fxi_node = n # Attach node to val
        return val

def fx_add_shapes(f_trace : torch.fx.GraphModule, sample_input : Any):
    """
    Run shape propagation on graph `f_trace`, which will add shape metadata in place.
    """
    torch.fx.passes.graph_manipulation.ShapeProp(f_trace).run(sample_input)

def fx_shape(x):
    """
    Return the shape of FX Proxy x.

    Assumes that ShapeProp has been run on the graph, so that x.fsi_node is set
    """
    return x.fxi_node.meta['tensor_meta'].shape


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
            assert kwargs == None or len(kwargs) == 0

            if target not in ad_map:
                raise NotImplementedError(f"Need VJP rule for {target}")
            # Look up forward/backward functions in `ad_map`
            fwd,bwd = ad_map[target]
            # Call the fwd function, getting proxies for returns
            val,aux = fwd(*args)
            # In the backward pass, we will compute:
            #  d[args[0]],...,d[args[-1]] = bwd(aux, d{val})
            # So remember: (args, bwd, aux, val)
            # Note that all of these are symbolic, so it's cheap to remember them
            self.stack.append((args, bwd, aux, val))
            # And return the return value (a proxy)
            return val

        def call_method(self, target, args, kwargs):
            raise NotImplementedError # use method_to_function

        def get_attr(self, target, args, kwargs):
            raise NotImplementedError # TODO

    # Grab the FX graph
    f_trace = torch.fx.symbolic_trace(f)
    
    # Run shape analysis, record answers in the graph
    fx_add_shapes(f_trace, sample_input)

    # This is the "template" function for the VJP
    def vjp_template(x, dret):
        # Run the forward computations, and collect them in ad.stack
        ad = ADInterpreter(f_trace)
        ret = ad.run(x)
        # Build a dict to hold derivatives
        d =  defaultdict(lambda: 0)
        # Add dret to derivatives dict
        d[ret] = dret
        # And run down the stack...
        for (args, bwd, aux, val) in reversed(ad.stack):
            dargs = bwd(aux, d[val])
            for (a,da) in zip(args, ensure_tuple(dargs)):
                d[a] += da
        # And return ret and J'*dret
        return ret, d[x]

    # Trace through vjp_template and return.
    return torch.fx.symbolic_trace(vjp_template)



import vjp_rules

def vjp_linear(f):
    """
    Construct fwd and bwd for f a linear function of x
    """
    def fwd(*args): return f(*args), None
    def bwd(_, dret): return f(dret)
    return fwd, bwd

import operator
ad_map[operator.neg] = vjp_linear(operator.neg)
ad_map[operator.add] = (vjp_rules.add_fwd, vjp_rules.add_bwd),
ad_map[operator.mul] = (vjp_rules.mul_fwd, vjp_rules.mul_bwd),
ad_map[operator.matmul] = (vjp_rules.matmul_fwd, vjp_rules.matmul_bwd),
ad_map[torch.neg] = vjp_linear(torch.neg),
ad_map[torch.sin] = {vjp_rules.sin_fwd, vjp_rules.sin_bwd},
ad_map[torch.relu] = (vjp_rules.relu_fwd, vjp_rules.relu_bwd),
ad_map[torch.transpose] = (vjp_rules.transpose_fwd, vjp_rules.transpose_bwd),
ad_map[torch.diag] = (vjp_rules.diag_fwd, vjp_rules.diag_bwd),
ad_map[vjp_rules.scale] = (vjp_rules.scale_fwd, vjp_rules.scale_bwd),

# Custom pass for trace (as we needed to use fx_shape)
def fx_trace_fwd(x):
    assert len(fx_shape(x)) == 2
    return torch.trace(x), fx_shape(x)
ad_map[torch.trace] = (fx_trace_fwd, vjp_rules.trace_bwd)

# And let's add some shape checking to add and mul, as we have it...
def fx_add_fwd(A,B):
    assert fx_shape(A) == fx_shape(B)
    return vjp_rules.add_fwd(A,B)
ad_map[operator.add] = (fx_add_fwd, vjp_rules.add_bwd)

def fx_mul_fwd(A,B):
    assert fx_shape(A) == fx_shape(B)
    return vjp_rules.mul_fwd(A,B)
ad_map[operator.mul] = (fx_mul_fwd, vjp_rules.mul_bwd)

