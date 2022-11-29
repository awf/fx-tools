from types import FunctionType, MethodDescriptorType
from typing import Tuple, Type, Callable, Dict, Union, Any
from dataclasses import dataclass
from icecream import ic

import torch
import torch.fx as tfx

from fx_print import fx_print_node

# --------------
def shape2str(sh: torch.Size):
    return ",".join([str(s) for s in sh])


def type2str(ty):
    return ty.__name__ if isinstance(ty, Type) else str(ty)


# --------------
@dataclass
class ShapeAndType:
    """
    A class representing the shape and type of a Python object.
    For classes other than torch.Tensor, it's simply the type.
    For torch.Tensor, it's the type, shape, and dtype.

    For a large number of operations, the ShapeAndType of the output
    can be determined as a function of its inputs.
    """

    ty: Type
    sh: Tuple[int]  # Valid only if ty == Tensor else ()
    dty: torch.dtype  # Valid only if ty == Tensor else None

    def __str__(self):
        if self.isTensor:
            return f"shnty:Tensor[({shape2str(self.sh)}),{type2str(self.dty)}]"
        else:
            return f"shnty[{type2str(self.ty)}]"

    def __repr__(self):
        return str(self)

    @property
    def isTensor(self) -> bool:
        return self.ty == torch.Tensor

    def dtype_or_type(self):
        return self.dty if self.isTensor else self.ty

    def dtype_or_default_type(self):
        if self.isTensor:
            return self.dty

        if isinstance(self.ty, torch.dtype):
            return self.ty
        elif self.ty in (float,):
            return torch.get_default_dtype()
        elif self.ty in (int,):
            return torch.int32

        raise NotImplementedError(f"Default dtype for {self.ty}")

    def ok(self) -> bool:
        if not isinstance(self.ty, (Type, torch.dtype)):
            return False
        if self.isTensor:
            if not isinstance(self.sh, tuple):
                return False
            if not all(isinstance(s, int) for s in self.sh):
                return False
            if not isinstance(self.dty, torch.dtype):
                return False
        return True


def shnty_ok(s):
    return isinstance(s, ShapeAndType) and s.ok()


def get_shnty(x):
    """
    Construct a ShapeAndType from an object x
    """
    if isinstance(x, torch.Tensor):
        ty = torch.Tensor
        sh = tuple(s for s in x.shape)
        dty = x.dtype
    else:
        ty = type(x)
        sh = ()
        dty = None

    return ShapeAndType(ty, sh, dty)


# This is the mapping from (function|method) to shape propagator
FunctionSpec = FunctionType  # Function, e.g. torch.relu, operator.mul
MethodSpec = Tuple[Type, str]  # Method, e.g. (torch.Tensor, 'neg')
ModuleSpec = Any  # TODO: modules
ShapePropagator = Callable[..., ShapeAndType]

global _shnty_propagator_dict
_shnty_propagator_dict: Dict[
    Union[FunctionSpec, MethodSpec, ModuleSpec], ShapePropagator
] = {}


def shnty_propagator_add(op, propagator):
    assert op not in _shnty_propagator_dict
    _shnty_propagator_dict[op] = propagator


def shnty_propagator(op):
    """
    Decorator to declare a function as a ShapeAndType propagator.

    ```
    @shnty_propagator(OP)
    def _(*ARGS):
      return ShapeAndType of OP applied to ARGS
    ```

    For global functions, OP is just the function.
    For methods, it is the pair (Type, method), e.g. (torch.Tensor, "sin")

    For example, a basic propagator for matmul (ignoring promotion, broadcast, etc)
    might look like

    ```
    @shnty_propagator(operator.matmul)
    def _(x, y):
      return ShapeAndType(x.ty, (x.sh[0], y.sh[1]), x.dty)
    ```
    """

    def the_decorator(the_propagator):
        shnty_propagator_add(op, the_propagator)
        return the_propagator  # normally just gets assigned to '_'

    return the_decorator


# ================= Use ShapeAndType in FX tracing ================
class ShapeAndTypeProxy(tfx.Proxy):
    TAG = "shnty"

    def __init__(self, node, tracer, shnty):
        super().__init__(node, tracer)
        assert self.TAG not in node.meta
        node.meta[self.TAG] = shnty

        # TODO: Be careful about namespace pollution here - a Proxy could be anything
        # e.g. a thing which has a ".node_" field...
        self.shnty_node_ = node

    @property
    def shape(self):
        shnty = self.shnty_node_.meta[self.TAG]
        assert shnty.isTensor
        return shnty.sh

    @property
    def T(self):
        shnty = self.shnty_node_.meta[self.TAG]
        assert shnty.isTensor
        return self.t()


def fx_get_shnty(x):
    if isinstance(x, tfx.Node):
        if ShapeAndTypeProxy.TAG not in x.meta:
            print(x.graph)
            raise RuntimeError(f"This node {x} is not from a shnty_trace?")
        return x.meta[ShapeAndTypeProxy.TAG]

    if isinstance(x, tfx.Proxy):
        return x.shnty_node_.meta[ShapeAndTypeProxy.TAG]

    # It's not an FX object.
    return get_shnty(x)


def get_type(x):
    # Could be more effiicient, but let's wait until settled
    return fx_get_shnty(x).ty


def justone(iter):
    val = tuple(iter)
    assert len(val) == 1
    return val[0]


def fn_name(f):
    n = f.__name__
    if hasattr(f, "__module__"):
        return f"{f.__module__}.{n}"
    else:
        return n


log = lambda x: print(x)


class ShapeAndTypeTracer(tfx.Tracer):
    def __init__(self, arg_shntys):
        super().__init__()
        self.arg_shntys_used = 0
        self.arg_shntys = arg_shntys

    def proxy(self, node):
        log(f"shnty_trace -- {fx_print_node(node)}")
        if node.op == "placeholder":
            if self.arg_shntys_used >= len(self.arg_shntys):
                raise ValueError("Not enough arg_shntys passed to shnty_trace")
            shnty = self.arg_shntys[self.arg_shntys_used]
            self.arg_shntys_used += 1

        elif node.op in ("call_function", "call_method"):
            # Get arg shntys
            arg_shntys = tuple(fx_get_shnty(self._inline_const(x)) for x in node.args)

            # Make lookup key
            if node.op == "call_function":
                key = node.target
                key_str = f"function `{fn_name(key)}`"
            elif node.op == "call_method":
                obj_ty = arg_shntys[0].ty
                key = (obj_ty, node.target)
                key_str = f"method `{type2str(obj_ty)}.{node.target}`"
            else:
                assert False

            if key not in _shnty_propagator_dict:
                raise NotImplementedError(
                    f"Need to implement `shnty_propagate` for {key_str}"
                )

            # Call the propagator
            shnty = _shnty_propagator_dict[key](*arg_shntys)
            if not shnty_ok(shnty):
                raise ValueError(
                    f"shnty_propagate[{key_str}] returned {shnty},"
                    + " which is not a valid instance of `ShapeAndType`"
                )
        else:
            raise NotImplementedError(f"ShapeAndTypeTracer proxy for {node.op}")

        return ShapeAndTypeProxy(node, self, shnty)

    def _inline_const(self, arg):
        if isinstance(arg, tfx.Node) and arg.op == "get_attr":
            return justone(k for (k, v) in self.tensor_attrs.items() if v == arg.target)
        else:
            return arg

    # def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: Dict[str, Any]):
    #     ic(attr, attr_val)
    #     return super().getattr(attr, attr_val, parameter_proxy_cache)


def shnty_trace(func, arg_shntys):
    log(f"shnty_trace {arg_shntys}")
    shnty_tracer = ShapeAndTypeTracer(arg_shntys)
    graph = shnty_tracer.trace(func)
    return tfx.GraphModule(shnty_tracer.root, graph, func.__name__)


def get_return_shnty(gm: tfx.GraphModule):
    outputs = [n for n in gm.graph.nodes if n.op == "output"]
    assert len(outputs) == 1
    for n in reversed(gm.graph.nodes):
        assert n.op == "output"
        return fx_get_shnty(n.args[0])
