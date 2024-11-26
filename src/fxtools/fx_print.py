import operator
import re
import torch
import types

from colorama import Fore, Back, Style

from awfutils import ndarray_str


def _commajoin(vs):
    return ",".join(vs)


def fn_name(f):
    n = f.__name__
    if hasattr(f, "__module__") and f.__module__ != "_operator":
        if f.__module__ == "torch._ops.aten":
            n = f"aten.{n}"
        else:
            n = f"{f.__module__}.{n}"

    # if hasattr(f, "__code__"):
    #     n += f"[{f.__code__.co_filename}:{f.__code__.co_firstlineno}]"

    return n


_default_ignore = {"creation_timestamp", "stack_trace", "type", "original_node"}


def fx_print_node(node, gm=None, name2ord=None, ignore=_default_ignore):
    def namestr(a):
        return name2ord[a.name] if name2ord else a.name

    def argstr(a):
        if isinstance(a, tuple):
            return "(" + _commajoin(map(argstr, a)) + ")"
        if isinstance(a, (list, torch.fx.immutable_collections.immutable_list)):
            return "[" + _commajoin(map(argstr, a)) + "]"
        if isinstance(a, torch.fx.Node):
            # Peek to see if it's a scalar constant
            if a.op == "get_attr" and len(a.args) == 0 and gm and hasattr(gm, a.target):
                val = getattr(gm, a.target)
                if val.shape == ():
                    return f"{val}"

            return namestr(a)
        if isinstance(a, (float,)):
            return f"{type(a).__name__}({a})"
        if isinstance(a, str):
            return f"'{a}'"

        # explicit list of types for which 'repr' is appropriate
        if isinstance(a, (int, torch.dtype, types.NoneType)):
            return repr(a)

        # a type we haven't seen before: print as <class 'T'>val
        # fxpn is greppable for those who would like to improve the printing
        return f"fxpn{type(a)}{repr(a)}"

    def prshape(s: torch.Size):
        if len(s) == 0:
            return "()"
        else:
            return "x".join(str(s) for s in s)

    def value_str(v):
        if isinstance(v, torch.Tensor):
            return ndarray_str(v.numpy())  # TODO: optimize for super-large tensors
        return str(v)

    def print_tensor_meta(tm):
        if isinstance(tm, torch.fx.passes.shape_prop.TensorMetadata):
            return f"Tensor[{prshape(tm.shape)}, {tm.dtype}]"
        if isinstance(tm, tuple):
            return "(" + ",".join(print_tensor_meta(t) for t in tm) + ")"

        raise ValueError(f"Unknown tensor meta {tm}")

    print_meta_handlers = {
        # tensor_meta:TensorMetadata(shape=torch.Size([2, 32]), dtype=torch.float32, requires_grad=False, stride=(32, 1), memory_format=torch.contiguous_format, is_quantized=False, qparams={})
        "tensor_meta": print_tensor_meta,
        # "type": lambda v: v.__name__ if v not in (torch.Tensor, tuple) else "type",
    }

    argstrs = [argstr(a) for a in node.args]
    meta_strs = ",".join(
        [
            print_meta_handlers[k](v)
            for k, v in node.meta.items()
            if k in print_meta_handlers
        ]
        + [
            str(k) + ":" + str(v)
            for k, v in node.meta.items()
            if k not in ignore and k not in print_meta_handlers
        ]
    )
    comment = f"{meta_strs}" if len(meta_strs) > 0 else ""

    def com():
        if comment:
            return Fore.GREEN + " # " + comment + Style.RESET_ALL
        else:
            return ""

    if node.op == "output":
        return f"return {argstrs[0]}{com()}"

    lhs = namestr(node)
    if node.op == "placeholder":
        return f"{lhs} = {node.target}{com()}"

    if node.op == "call_function":
        if node.target == operator.getitem:
            # Silly sugar for getitem, but it's nicer to read...
            return f"{lhs} = {argstrs[0]}[{_commajoin(argstrs[1:])}]{com()}"
        else:
            if hasattr(node.target, "__code__"):
                comment += f"[{node.target.__code__.co_filename}:{node.target.__code__.co_firstlineno}]"

            return f"{lhs} = {fn_name(node.target)}({_commajoin(argstrs)}){com()}"

    if node.op == "call_method":
        return f"{lhs} = {argstrs[0]}.{node.target}({_commajoin(argstrs[1:])}){com()}"

    if node.op == "call_module":
        return f"{lhs} = {node.target}({_commajoin(argstrs)}){com()}"

    if node.op == "get_attr":
        assert len(argstrs) == 0
        if gm:
            if hasattr(gm, node.target):
                val = getattr(gm, node.target)
                valstr = value_str(val)
            else:
                valstr = f"no attr {node.target}"

            if val.size() == ():
                return f"# {lhs} = {val}"
            else:
                comment += valstr
                return f"{lhs} = self.{node.target}{com()}"

        return f"{lhs} = getattr({node.target}, {node.args})"

    return f"# unhandled {node.op} {lhs} = {node.target}({_commajoin(argstrs)}){com()}"


def fx_print_iter(gm, ignore=_default_ignore):

    name2ord = {}
    ord = 10

    args = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
    name = torch.nn.Module._get_name(gm)
    yield f"def {name}({_commajoin(args)}):"

    for m, val in gm.named_modules():
        if m != "":
            yield f"  {m} = {val.__module__}.{val}"

    for node in gm.graph.nodes:
        assert node.name not in name2ord
        name2ord[node.name] = f"v{ord}"
        ord += 1

        yield "  " + fx_print_node(node, gm, name2ord, ignore)


def fx_print(gm, ignore=_default_ignore):

    for line in fx_print_iter(gm, ignore):
        print(line)


def test_fx_print():
    # Just a non-crash test
    # TODO: code round-trip test
    def aux(p, q):
        return torch.relu(1.234 * p * q).neg()

    def my_func(x, b):
        y = 2 * x
        for _ in range(b):  # Loops will be unrolled
            x = aux(x, y)  # Function calls will be inlined
        x = x.neg()
        x = x @ torch.eye(3)
        x = torch.transpose(x, 1, 0)
        x = torch.sum(x, [0, 1])

        return torch.atan2(torch.tensor(float(b)), x)

    my_func(torch.rand(3, 3), 5)

    def compiler_fn(fx_module: torch.fx.GraphModule, example_inputs):
        fx_print(fx_module)
        return fx_module

    aux_gm = torch.compile(aux, backend=compiler_fn)

    aux_gm(torch.rand(3, 3, device="meta"), torch.rand(3, 3, device="meta"))

    # Pass on the compiler_fn to the aot_function API
    my_func_gm = torch.compile(my_func, backend=compiler_fn)

    my_func_gm(torch.rand(3, 3), 5)

    print("done")


def fx_print_to_file(gm, filename):
    print(f"fx_print_to_file: saving to {filename}")
    with open(filename, "w") as f:
        print(
            f"""# Autogenerated from {__file__}
import torch
aten = torch.ops.aten
inf = float('inf')
""",
            file=f,
        )
        for line in fx_print_iter(
            gm, ignore={"val", "nn_module_stack", "creation_timestamp", "stack_trace"}
        ):
            print(line, file=f)


if __name__ == "__main__":
    test_fx_print()
