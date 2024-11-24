# FX Tools

## Experiments with PyTorch FX

This is a small library of tools for dealing with FX graphs, and some example 
transformations.

Of most interest is probably the source-to-source automatic differentiation code in 
`difffx.py`, and its even simpler development at https://github.com/awf/fx-tools/blob/main/DiffFX_Derivation.ipynb.  Its essence is in a few dozen lines of code:
```py
class ADInterpreter(torch.fx.Interpreter):
  ...
    def call_function(self, target, args, kwargs):
        # Look up forward/backward functions in `ad_map`
        fwd,bwd = ad_map[target]
        # Call the fwd function, getting proxies for returns
        val,aux = fwd(*args)
        # Record the call and arguments (symbolically) for the backward pass
        self.stack.append((args, bwd, aux, val))
  ...

def vjp_template(x, dret):
    # Build a dict to hold derivatives
    d =  defaultdict(lambda: 0)
    # Add dret to derivatives dict
    d[ret] = dret
    # And run down the stack...
    for (args, bwd, aux, val) in reversed(ad.stack):
        # Call to "bwd"
        dargs = bwd(aux, d[val])
        # And update the derivatives wrt the args
        for (a,da) in zip(args, dargs):
            d[a] += da
    # And return ret and J'*dret
    return ret, d[x]
```
Although the above might look like it's actually computing the derivatives, 
it's a template that is traced through `torch.fx.symbolic_trace` to give a 
source-to-source implementation.

For example, this input code (from https://github.com/awf/fx-tools/blob/main/DiffFX_Experiments.ipynb)
```py
def f(x):
    return torch.sin(x) + x
```
Produces
```py
def vjp_template(x,dret):
  v10 = x
  v11 = dret
  v12 = torch.sin(v10)
  v13 = add(v12,v10)
  v14 = v11.reshape((13))
  v15 = v11.reshape((13))
  v16 = add(0,v14)
  v17 = add(0,v15)
  v18 = torch.cos(v10)
  v19 = mul(v18,v16)
  v20 = add(v17,v19)
  return (v13,v20)
```

Which one can hand-optimize (TODO: implement some of these simple optimizations) to
```py
def vjp_template(x,dret):
  v12 = torch.sin(x) + x
  v14 = dret.reshape((13))
  v18 = torch.cos(x)
  v19 = mul(v18,v14)
  v20 = add(v14,v19)
  return (v13,v20)
```


In contrast, torch's implementation is lower-level, so produces (arguably)
less "hackable" code:
```py
def f(x_1):
  v10 = x_1
  v11 = aten.sin.default(v10)
  v12 = aten.add.Tensor(v11,v10)
  v13 = self._tensor_constant0
  v14 = aten.lift_fresh_copy.default(v13)
  v15 = aten.cumsum.default(v14,0)
  v16 = aten.slice.Tensor(v15,0,0,-1)
  v17 = aten.neg.default(v16)
  v18 = aten.unbind.int(v17)
  v19 = aten.new_zeros.default(v12,[13,13])
  v20 = aten.diagonal.default(v19)
  v21 = aten.fill_.Scalar(v20,1)
  v22 = aten.view.default(v19,[13,13])
  v23 = aten.cos.default(v10)
  v24 = aten.mul.Tensor(v22,v23)
  v25 = aten.add.Tensor(v22,v24)
  v26 = aten.split_with_sizes.default(v25,[13])
  v27 = v26[0]
  v28 = aten.view.default(v27,[13,13])
  return v28
```

https://discuss.pytorch.org/t/torch-fx-vs-torch-jit-script/100299/3


## Shape propagation

This is an aside -- we need shapes for e.g. the gradient of 'trace', so let's
quickly assemble a solution, where we can call `fx_shape` on proxies.

Unlike ShapeProp, there is no FakeTensor subclass.  This is most likely a disadvantage, 
so this should be considered mor
https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern
    - good for memory/time
    - good because shnty_propagator explicitly knows if it's OK to propagate

Good that we can get type, e.g. to resolve call_method

important that we use getattr(t, "neg")() sometimes instead of torch.Tensor.neg(t) as the latter may defeat monkey patching


Quick hack here, as we expect more thorough handling upstream:
 - https://discuss.pytorch.org/t/fx-proxy-and-shapes/113861/4
 - https://github.com/pytorch/pytorch/issues/54982
 - https://www.youtube.com/watch?v=pLni96jtcjY

