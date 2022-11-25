from types import FunctionType
from typing import Tuple, Type, Callable, Dict, Union, Any
from dataclasses import dataclass

import torch
import torch.fx as tfx


# --------------
def scalar_shape():
    return ()


def is_scalar_shape(shape):
    return shape == scalar_shape()


def type2str(ty):
    return (f"{ty.__module__}." if ty.__module__ else "") + (f"{ty.__name__}")


def shape2str(sh: torch.Size):
    return ",".join([str(s) for s in sh])


# --------------
@dataclass
class ShapeAndType:
    sh: torch.Size
    ty: Type

    def __str__(self):
        return f"{type2str(self.ty)}[{shape2str(self.sh)}]"


class ShapeAndTypeProxy(tfx.Proxy):
    TAG = "shnty"

    def __init__(self, node, tracer, shnty):
        super().__init__(node, tracer)
        assert self.TAG not in node.meta
        node.meta[self.TAG] = shnty

        # TODO: Be careful about namespace pollution here - a Proxy could be anything
        # e.g. a thing which has a ".node_" field...
        self.my_node_ = node

    @property
    def shape(self):
        print("shape")
        return self.my_node_.meta[self.TAG].sh


def get_shnty(x):
    if isinstance(x, tfx.Node):
        return x.meta[ShapeAndTypeProxy.TAG]

    if isinstance(x, tfx.Proxy):
        return x.my_node_.meta[ShapeAndTypeProxy.TAG]

    # It's not an FX object, just get type...
    ty = type(x)

    # ... and shape ...
    if isinstance(x, (int, float)):
        sh = scalar_shape()
    else:
        sh = x.shape

    # ... and make ShapeAndType
    return ShapeAndType(sh, ty)


def get_type(x):
    # Could be more effiicient, but let's wait until settled
    return get_shnty(x).ty


# This is the mapping from (function|method) to shape propagator
FunctionSpec = FunctionType  # Function, e.g. torch.relu, operator.mul
MethodSpec = Tuple[Type, str]  # Method, e.g. (torch.Tensor, 'neg')
ModuleSpec = Any  # TODO: modules
ShapePropagator = Callable[..., ShapeAndType]

shnty_propagate: Dict[Union[FunctionSpec, MethodSpec, ModuleSpec], ShapePropagator] = {}


class ShapeAndTypeTracer(tfx.Tracer):
    def __init__(self, arg_shntys):
        super().__init__()
        self.arg_shntys_used = 0
        self.arg_shntys = arg_shntys

    def proxy(self, node):
        if node.op == "placeholder":
            if self.arg_shntys_used >= len(self.arg_shntys):
                raise ValueError("Not enough arg_shntys passed to shnty_trace")
            shnty = self.arg_shntys[self.arg_shntys_used]
            self.arg_shntys_used += 1

        elif node.op == "call_function":
            arg_shntys = tuple(get_shnty(x) for x in node.args)
            shnty = shnty_propagate[node.target](*arg_shntys)

        elif node.op == "call_method":
            arg_shntys = tuple(get_shnty(x) for x in node.args)
            obj_ty = get_type(node.args[0])
            shnty = shnty_propagate[(node.target, obj_ty)](*arg_shntys)
        else:
            raise NotImplemented(f"ShapeAndTypeTracer proxy for {node.op}")

        return ShapeAndTypeProxy(node, self, shnty)


def shnty_trace(func, arg_shntys):
    tracer = ShapeAndTypeTracer(arg_shntys)
    graph = tracer.trace(func)
    return tfx.GraphModule(tracer.root, graph, func.__name__)


# --------------
import operator


def broadcast_shapes(arg1, *args):
    if len(args) == 0:
        return arg1
    else:
        return torch.broadcast_shapes(arg1, *args)


def broadcast_shnty(arg1, *args):
    sh = broadcast_shapes(arg1.sh, *(arg.sh for arg in args))
    ty = torch.Tensor if not is_scalar_shape(sh) else type(arg1)
    return ShapeAndType(sh, ty)


# assert A_shape[1] == B_shape[0]
# return A_shape[0], B_shape[1]

# TODO: not names, ops:
shnty_propagate[operator.mul] = broadcast_shnty
shnty_propagate[operator.neg] = broadcast_shnty
shnty_propagate[("neg", torch.Tensor)] = broadcast_shnty
shnty_propagate[torch.relu] = broadcast_shnty
shnty_propagate[torch.atan2] = broadcast_shnty

# --------------
def test_shnty():
    from fx_print import fx_print

    def aux(p, q):
        return torch.relu(1.234 * p * q).neg()

    def my_func(x, b):
        y = 2 * x
        for _ in range(2):  # Loops will be unrolled
            x = aux(x, y)  # Function calls will be inlined
        return torch.atan2(b, x)

    x_shnty = ShapeAndType((3, 5), torch.Tensor)
    b_shnty = ShapeAndType((), int)
    gm = shnty_trace(my_func, arg_shntys=(x_shnty, b_shnty))

    fx_print(gm)


if __name__ == "__main__":
    test_shnty()
