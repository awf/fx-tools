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

    def assert_ok(self, msg):
        if not isinstance(self.ty, (Type, torch.dtype)):
            raise ValueError(msg, f"ty {self.ty} not a type")
        if not self.isTensor:
            if self.sh != ():
                raise ValueError(msg, f"Non-tensor has non-empty shape {self.sh}")
            return
        if not isinstance(self.sh, tuple):
            raise ValueError(msg, f"shape {self.sh} not a tuple")
        if not all(isinstance(s, int) for s in self.sh):
            raise ValueError(msg, f"shape {self.sh} not all ints")
        if not isinstance(self.dty, torch.dtype):
            raise ValueError(msg, f"dtype {self.dty} not a torch.dtype")


def assert_shnty_ok(msg, s):
    if not isinstance(s, ShapeAndType):
        raise ValueError(msg, f"Not a ShapeAndType: {s}")
    s.assert_ok(msg)


def shnty_from_val(x):
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
    """shnty_propagator_add(op, propagator):
    Register PROPAGATOR as tbe peopagtor for OP.

    See `shnty_propagator`
    """
    assert op not in _shnty_propagator_dict
    _shnty_propagator_dict[op] = propagator


def _fn_name(f):
    n = f.__name__
    if hasattr(f, "__module__"):
        return f"{f.__module__}.{n}"
    else:
        return n


def _opspec_to_str(opspec):
    if isinstance(opspec, Callable):
        return f"function `{_fn_name(opspec)}`"

    if isinstance(opspec, tuple):
        ty, attr = opspec
        assert isinstance(ty, Type)
        assert isinstance(attr, str)
        return f"method `{type2str(ty)}.{attr}`"

    assert False


def shnty_propagator(opspec):
    """
    Decorator to declare a function as a ShapeAndType propagator.

    That is, if OP is a global function
      op: *S -> T
    then
      propagator: *ShapeAndType -> ShapeAndType
    such that
      get_shnty(op(s)) == propagator(get_shnty(s))

    ```
    @shnty_propagator(OP)
    def _(*ARGS):
      return ShapeAndType of OP applied to ARGS
    ```

    The argument OPSPEC is a unique key identifying OP.
    For a global function, e.g. `opertor.matmul` or `torch.sin`,
    it is the function itself.
    For methods, it is the pair (Type, method), e.g. (torch.Tensor, "sin")

    For example, a basic propagator declaration for matmul (ignoring promotion,
    broadcast, etc) might look like
    ```
    @shnty_propagator(operator.matmul)
    def _(x, y):
      return ShapeAndType(x.ty, (x.sh[0], y.sh[1]), x.dty)
    ```

    And for transpose (again ignoring broadcast etc), might look like
    ```
    @shnty_propagator((torch.Tensor, "t"))
    def _(x):
      return ShapeAndType(x.ty, (x.sh[1], x.sh[0]), x.dty)
    ```


    """

    def the_decorator(the_propagator):
        shnty_propagator_add(opspec, the_propagator)
        # Override name, normally just '_'
        the_propagator.__name__ = (
            f"{the_propagator.__name__}<shnty_propagator({_opspec_to_str(opspec)})>"
        )
        the_propagator.__qualname__ = (
            the_propagator.__module__ + "." + the_propagator.__name__
        )

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
    def _shnty(self):
        return self.shnty_node_.meta[self.TAG]

    # Provide torch.Tensor properties
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
    return shnty_from_val(x)


def fx_get_shnty_or_val(x):
    if isinstance(x, tfx.Node):
        if ShapeAndTypeProxy.TAG not in x.meta:
            print(x.graph)
            raise RuntimeError(f"This node {x} is not from a shnty_trace?")
        return x.meta[ShapeAndTypeProxy.TAG]

    if isinstance(x, tfx.Proxy):
        return x.shnty_node_.meta[ShapeAndTypeProxy.TAG]

    # It's not an FX object.
    # Return shnty if Tensor, else val
    if isinstance(x, torch.Tensor):
        return shnty_from_val(x)

    return x


def fx_shape(x):
    return fx_get_shnty(x).sh


def justone(iter):
    val = tuple(iter)
    assert len(val) == 1
    return val[0]


_log = lambda x: print(x)


class ShapeAndTypeTracer(tfx.Tracer):
    def __init__(self, arg_shntys):
        super().__init__()
        self.arg_shntys_used = 0
        self.arg_shntys = arg_shntys

    def proxy(self, node):
        _log(f"shnty_trace -- {fx_print_node(node)}")
        if node.op == "placeholder":
            if self.arg_shntys_used >= len(self.arg_shntys):
                raise ValueError("Not enough arg_shntys passed to shnty_trace")
            shnty = self.arg_shntys[self.arg_shntys_used]
            self.arg_shntys_used += 1

        elif node.op in ("call_function", "call_method"):
            # Get arg shntys
            arg_shntys_or_vals = tuple(
                fx_get_shnty_or_val(self._inline_const(x)) for x in node.args
            )

            # Make lookup key
            if node.op == "call_function":
                key = node.target
            elif node.op == "call_method":
                key = (arg_shntys_or_vals[0].ty, node.target)
            else:
                assert False

            if key not in _shnty_propagator_dict:
                raise NotImplementedError(
                    f"Need to implement `shnty_propagate` for {_opspec_to_str(key)}"
                )

            # Call the propagator
            shnty = _shnty_propagator_dict[key](*arg_shntys_or_vals)
            assert_shnty_ok(f"shnty_propagate[{_opspec_to_str(key)}]", shnty)
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
    _log(f"shnty_trace {arg_shntys}")
    shnty_tracer = ShapeAndTypeTracer(arg_shntys)
    graph = shnty_tracer.trace(func)
    return tfx.GraphModule(shnty_tracer.root, graph, func.__name__)


def get_return_shnty(gm: tfx.GraphModule):
    outputs = [n for n in gm.graph.nodes if n.op == "output"]
    assert len(outputs) == 1
    for n in reversed(gm.graph.nodes):
        assert n.op == "output"
        return fx_get_shnty(n.args[0])
