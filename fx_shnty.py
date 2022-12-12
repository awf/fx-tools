from types import FunctionType, MethodDescriptorType, GetSetDescriptorType
from typing import Tuple, Type, Callable, Dict, Union, Any
from dataclasses import dataclass
from icecream import ic
import numpy
import torch
import torch.fx as tfx

from fx_print import fx_print_node, fn_name

# --------------
def shape2str(sh: torch.Size):
    return ",".join([str(s) for s in sh])


def type2str(ty):
    return ty.__qualname__ if isinstance(ty, Type) else str(ty)


# --------------
@dataclass
class AbstractValue:
    """
    A class representing just the type of a Python object.

    For a large number of operations, the type of the output
    can be determined as a function of its inputs, using propagators
    defined using `shnty_propagator`

    The term 'Abstract', although heavily overloaded in computer science, comes
    from abstract interpretation (see https://en.wikipedia.org/wiki/Abstract_interpretation).
    """

    ty: Type

    def __str__(self):
        return f"abval[{type2str(self.ty)}]"

    def __repr__(self):
        return str(self)

    def __instancecheck__(self, ty):
        return isinstance(self.ty, ty)

    @property
    def isTensor(self) -> bool:
        return self.ty == torch.Tensor

    def assert_ok(self, msg):
        if not isinstance(self.ty, (Type, torch.dtype)):
            raise ValueError(msg, f"ty {self.ty} not a type")


@dataclass
class AbstractTensor(AbstractValue):
    """
    A class representing the the shape and dtype of a Python object.
    Numerous methods on Tensor are not supported by AbstractTensor,
    but several very useful ones are:
      shape
      dtype

    For a large number of operations, the AbstractValue of the output
    can be determined as a function of its inputs, using propagators
    defined using `shnty_propagator`
    """

    shape: torch.Size
    dtype: torch.dtype
    is_nested: bool

    # Implenemt all tensor methods that can be implemented using only shape and dtype
    def dim(self):
        return len(self.shape)

    def size(self, dim=None):
        if dim:
            return self.shape[dim]
        else:
            return self.shape

    @property
    def ndim(self):
        return self.dim()

    def numel(self):
        return numpy.prod(self.shape)

    def nelement(self):
        return self.numel()

    # Housekeeping for AbstractValue
    def __init__(self, shape, dtype):
        super().__init__(torch.Tensor)
        assert isinstance(shape, torch.Size)
        assert isinstance(dtype, torch.dtype)
        self.shape = shape
        self.dtype = dtype
        self.is_nested = False

    def __str__(self):
        return f"AbTensor[({shape2str(self.shape)}),{type2str(self.dtype)}]"

    def __repr__(self):
        return str(self)

    def assert_ok(self, msg):
        if not all(isinstance(s, int) for s in self.shape):
            raise ValueError(msg, f"shape {self.shape} not all ints")
        if not isinstance(self.dtype, torch.dtype):
            raise ValueError(msg, f"dtype {self.dtype} not a torch.dtype")


def assert_shnty_ok(msg, s):
    if not isinstance(s, AbstractValue):
        raise ValueError(msg, f"Not a AbstractValue: {s}")
    s.assert_ok(msg)


def abstractify(x):
    """
    Construct an AbstractValue from an object x
    """
    if isinstance(x, torch.Tensor):
        return AbstractTensor(x.shape, x.dtype)
    else:
        if isinstance(x, (float, int)):
            ty = torch.tensor(x).dtype
        else:
            ty = type(x)
        return AbstractValue(ty)


# This is the mapping from (function|method) to shape propagator
FunctionSpec = FunctionType  # Function, e.g. torch.relu, operator.mul
MethodSpec = Tuple[Type, str]  # Method, e.g. (torch.Tensor, 'neg')
ModuleSpec = Any  # TODO: modules
ShapePropagator = Callable[..., AbstractValue]

global _shnty_propagator_dict
_shnty_propagator_dict: Dict[
    Union[FunctionSpec, MethodSpec, ModuleSpec], ShapePropagator
] = {}


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


def shnty_propagator_register(op, propagator):
    """shnty_propagator_register(op, propagator):
    Register PROPAGATOR as tbe peopagtor for OP.

    See `shnty_propagator`
    """
    if isinstance(op, MethodDescriptorType):
        op = (_canonicalize_class(op.__objclass__), op.__name__)

    if op in _shnty_propagator_dict:
        raise RuntimeError(f"Shnty op {_opspec_to_str(op)} already registered")

    _shnty_propagator_dict[op] = propagator


def shnty_propagator(opspec):
    """
    Decorator to declare a function as a AbstractValue propagator.

    That is, if OP is a global function
      op: *S -> T
    then
      propagator: *AbstractValue -> AbstractValue
    such that
      get_shnty(op(s)) == propagator(get_shnty(s))

    ```
    @shnty_propagator(OP)
    def _(*ARGS):
      return AbstractValue of OP applied to ARGS
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
      x_shape = fx_shape(x)
      y_shape = fx_shape(y)
      return AbstractTensor((x_shape[0], y_shape[1]), x.dtype)
    ```

    And for transpose (again ignoring broadcast etc), might look like
    ```
    @shnty_propagator(torch.Tensor.T)
    def _(x):
      x_shape = fx_shape(x)
      return AbstractTensor((x_shape[1], x_shape[0]), x.dtype)
    ```


    """

    def the_decorator(the_propagator):
        shnty_propagator_register(opspec, the_propagator)
        # Override name, normally just '_'
        if the_propagator.__name__ == "_":
            the_propagator.__name__ = (
                f"{the_propagator.__name__}<shnty_propagator({_opspec_to_str(opspec)})>"
            )
            the_propagator.__qualname__ = (
                the_propagator.__module__ + "." + the_propagator.__name__
            )

        return the_propagator  # normally just gets assigned to '_'

    return the_decorator


# ================= Use AbstractValue in FX tracing ================
_AVP_TAG = "$abval"


class AbstractValueProxy(tfx.Proxy):
    # TODO: Be careful about namespace pollution here - a Proxy could be anything
    # e.g. a thing which has a ".node_" field...

    def __init__(self, node, tracer, abval):
        super().__init__(node, tracer)
        assert _AVP_TAG not in node.meta
        node.meta[_AVP_TAG] = abval

        self._abval_node = node

    def __str__(self):
        return f"AbValProxy[{self._abval_node.meta[_AVP_TAG]}]"

    def __getattr__(self, attr):
        # Forward to abval
        if hasattr(self._abval_node.meta[_AVP_TAG], attr):
            return getattr(self._abval_node.meta[_AVP_TAG], attr)
        return torch.fx.proxy.Attribute(self, attr)

    # See ParameterProxy for alternative list of forwarding methods:
    # The problem with this list is that it strongly assumes the proxy is for
    # a tensor
    # @property
    # def shape(self):
    #     return self.param.shape

    # def size(self):
    #     return self.param.size()

    # def dim(self):
    #     return self.param.dim()

    # @property
    # def ndim(self):
    #     return self.param.ndim

    # def numel(self):
    #     return self.param.numel()

    # def nelement(self):
    #     return self.param.nelement()

    def __instancecheck__(self, ty):
        return isinstance(self.ty, ty)

    @classmethod
    def __torch_function__(cls, orig_method, types, args=None, kwargs=None):
        # # Dispatch through some known modules
        # if orig_method == torch.nn.functional.multi_head_attention_forward:
        #     torch.overrides.has_torch_function_skips += 1
        #     return orig_method(*args, **kwargs)

        return super().__torch_function__(orig_method, types, args, kwargs)


def fx_get_abstract_value_or_value(x):
    if isinstance(x, tfx.Node):
        if _AVP_TAG not in x.meta:
            print(x.graph)
            raise RuntimeError(f"This node {x} is not from a shnty_trace?")
        return x.meta[_AVP_TAG]

    if isinstance(x, tfx.Proxy):
        return fx_get_abstract_value_or_value(x._abval_node)

    # It's not an FX object.
    return x


# Shape
def fx_shape(a):
    """
    Get shape of an object which might be a torch Tensor, or an AbstractTensor,
    or a scalar.
    """
    a = fx_get_abstract_value_or_value(a)

    if isinstance(a, (AbstractTensor, torch.Tensor)):
        return a.shape
    return ()  # Assume scalar e.g. 2.2 * torch.rand(2,3)


# Shape
def fx_type(a):
    """
    Get shape of an object which might be an AbstractValue
    """
    a = fx_get_abstract_value_or_value(a)

    if isinstance(a, AbstractValue):
        return a.ty
    return type(a)


def fx_is_tensor(x):
    """
    Return true if x is a Tensor or a tensor Proxy
    """
    x = fx_get_abstract_value_or_value(x)

    return isinstance(x, torch.Tensor) or x.isTensor


def justone(iter):
    val = tuple(iter)
    assert len(val) == 1
    return val[0]


_log = lambda *args, **kwargs: ...
_log = print


class AbstractValueTracer(tfx.Tracer):
    def __init__(self, aargs):
        super().__init__()  # TODO: param_shapes_constant?
        self.aargs_used = 0
        self.aargs = aargs
        self.called_modules = {}

    def proxy(self, node):
        _log(f"shnty_trace -- {fx_print_node(node, _AVP_TAG)}", end="...")
        if node.op == "placeholder":
            if self.aargs_used >= len(self.aargs):
                raise ValueError("Not enough aargs passed to shnty_trace")
            abval = self.aargs[self.aargs_used]
            self.aargs_used += 1
            _log(f"{abval}")
            if isinstance(abval, AbstractValue):
                return AbstractValueProxy(node, self, abval)
            else:
                return abval

        if node.op in ("call_function", "call_method", "call_module"):
            # Get arg shntys
            aargs_or_vals = tuple(
                fx_get_abstract_value_or_value(self._inline_const(x)) for x in node.args
            )
            ty0 = fx_type(aargs_or_vals[0])

            # Make lookup key
            if node.op == "call_function":
                if node.target == getattr:
                    assert len(aargs_or_vals) == 2
                    attr = node.args[1]
                    assert isinstance(attr, str)
                    if attr == "shape":
                        raise NotImplementedError(
                            "To use shnty_trace, consider using `fx_shape(x)` instead of `x.shape`"
                        )

                    key = getattr(ty0, node.args[1])
                    assert isinstance(key, GetSetDescriptorType)
                    aargs_or_vals = aargs_or_vals[:1]  # drop attr
                else:
                    key = node.target
            elif node.op == "call_method":
                key = (ty0, node.target)
            elif node.op == "call_module":
                m = self.called_modules[node.target]
                key = type(m)
                aargs_or_vals = (m,) + aargs_or_vals
            else:
                assert False

            if key not in _shnty_propagator_dict:
                msg = f"Need to implement `shnty_propagate` for {_opspec_to_str(key)}"
                _log(f" error: {msg} ({node.args})")
                raise NotImplementedError(msg)

            # Call the propagator
            abval = _shnty_propagator_dict[key](*aargs_or_vals)
            assert_shnty_ok(f"shnty_propagate[{_opspec_to_str(key)}]", abval)
            _log(f"{abval}")
            return AbstractValueProxy(node, self, abval)

        if node.op == "get_attr":
            # This None will be filled in in getattr below
            abval = None
            _log(" made proxy")
            return AbstractValueProxy(node, self, abval)

        _log(" error!")
        raise NotImplementedError(f"AbstractValueTracer proxy for {node.op}")

    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        """ """
        module_qualified_name = self.path_of_module(m)
        if not self.is_leaf_module(m, module_qualified_name):
            return forward(*args, **kwargs)
        # Stash the module and pass on to FX implementation
        self.called_modules[module_qualified_name] = m
        return self.create_proxy("call_module", module_qualified_name, args, kwargs)

    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: Dict[str, Any]):
        # Call super, get a proxy if appropriate
        ret = super().getattr(attr, attr_val, parameter_proxy_cache)

        # Attach attr_val to the returned proxy
        if isinstance(ret, AbstractValueProxy):
            abval = abstractify(attr_val)
            print(f'Attaching to {attr}: {abval}')
            ret.node.meta[_AVP_TAG] = abval

        return ret

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return super().is_leaf_module(m, module_qualified_name)

    def _inline_const(self, arg):
        if (
            isinstance(arg, tfx.Node)
            and arg.op == "get_attr"
            and _AVP_TAG not in arg.meta
        ):

            if len(arg.args) != 0:
                raise NotImplementedError("OI")

            matches_in_tensor_attrs = [
                k for (k, v) in self.tensor_attrs.items() if v == arg.target
            ]
            if len(matches_in_tensor_attrs) == 1:
                return matches_in_tensor_attrs[0]

            if len(matches_in_tensor_attrs) > 1:
                raise RuntimeError("oiks?")

            print(f"where is {arg.target}")

        return arg


def shnty_trace(func, aargs):
    """
    Perform a symboic trace of callable FUNC, at the given arguments
    Where arguments are AbstractValues, the trace will propagate their shapes
    and types.
    Other arguments will be treated as constants, and folded into the
    resulting GraphModule.

    Example:

    ```
    def foo(x5 : torch.Tensor, lr : float, n: int):


    """
    name = func.__name__ if isinstance(func, FunctionType) else func.__class__.__name__
    _log(f"shnty_trace {name} at {aargs}")
    shnty_tracer = AbstractValueTracer(aargs)
    graph = shnty_tracer.trace(func)
    return tfx.GraphModule(shnty_tracer.root, graph, name)


def get_return_abstract_value(gm: tfx.GraphModule):
    """
    Given an FX GraphModule created by `shnty_trace`, return
    an AbstractValue representing the return value of the computation.
    """
    outputs = [n for n in gm.graph.nodes if n.op == "output"]
    assert len(outputs) == 1
    # Assuming the last node is the "output" node
    for n in reversed(gm.graph.nodes):
        assert n.op == "output"
        return fx_get_abstract_value_or_value(n.args[0])


import fx_shnty_propagators
