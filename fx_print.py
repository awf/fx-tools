import re
import torch


def _commajoin(vs):
    return ",".join(vs)


def fn_name(f):
    n = f.__name__
    if hasattr(f, "__module__") and f.__module__ != "_operator":
        if f.__module__ == 'torch._ops.aten':
          n = f"aten.{n}"
        else:
          n = f"{f.__module__}.{n}"

    if hasattr(f, "__code__"):
        n += f"[{f.__code__.co_filename}:{f.__code__.co_firstlineno}]"

    return n

_default_ignore = {"creation_timestamp", "stack_trace"}

def fx_print_node(node, gm=None, name2ord=None, ignore=_default_ignore):
    def argstr(a):
        if isinstance(a, tuple):
            return "(" + _commajoin(map(argstr, a)) + ")"
        if isinstance(a, (list, torch.fx.immutable_collections.immutable_list)):
            return "[" + _commajoin(map(argstr, a)) + "]"
        if isinstance(a, torch.fx.Node):
            return name2ord[a.name] if name2ord else a.name
        if isinstance(a, (float,)):
            return f"{type(a).__name__}({a})"
        if isinstance(a, int):
            return f"{a}"
        if isinstance(a, str):
            return f"'{a}'"

        return f"{type(a)}{repr(a)}"

    def prshape(s: torch.Size):
        return "x".join(str(s) for s in s)

    def value_str(v):
        vstr = str(v)
        vstr = re.sub(r"\s*\n\s*", r"\\n ", vstr)
        if len(vstr) > 40:
            vstr = vstr[:30] + "<...>" + vstr[-10:]
        if isinstance(v, torch.Tensor):
            vstr = f"Tensor[{prshape(v.shape)}, {v.dtype}]({vstr})"
        return vstr

    print_meta_handlers = {
        # tensor_meta:TensorMetadata(shape=torch.Size([2, 32]), dtype=torch.float32, requires_grad=False, stride=(32, 1), memory_format=torch.contiguous_format, is_quantized=False, qparams={})
        "tensor_meta": (lambda tm: f"Tensor[{prshape(tm.shape)}, {tm.dtype}]")
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
    comment = f" # {meta_strs}" if len(meta_strs) > 0 else ''

    if node.op == "output":
        return f"return {argstrs[0]}{comment}"

    lhs = argstr(node)
    if node.op == "placeholder":
        return f"{lhs} = {node.target}{comment}"

    if node.op == "call_function":
        return f"{lhs} = {fn_name(node.target)}({_commajoin(argstrs)}){comment}"

    if node.op == "call_method":
        return f"{lhs} = {argstrs[0]}.{node.target}({_commajoin(argstrs[1:])}){comment}"

    if node.op == "call_module":
        return f"{lhs} = {node.target}({_commajoin(argstrs)}){comment}"

    if node.op == "get_attr":
        assert len(argstrs) == 0
        if gm:
            if hasattr(gm, node.target):
                val = getattr(gm, node.target)
                valstr = value_str(val)
            else:
                valstr = f"no attr {node.target}"

            assert node.args == ()
            return f"{lhs} = self.{node.target} # {valstr}"

        return f"{lhs} = getattr({node.target}, {node.args})"

    return (
        f"# unhandled {node.op} {lhs} = {node.target}({_commajoin(argstrs)}){comment}"
    )


def fx_print_iter(gm, ignore=_default_ignore):

    name2ord = {}
    ord = 10

    args = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
    name = torch.nn.Module._get_name(gm)
    yield f"def {name}({_commajoin(args)}):"

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

    aux_gm(torch.rand(3, 3, device='meta'), torch.rand(3,3, device='meta'))

    # Pass on the compiler_fn to the aot_function API
    my_func_gm = torch.compile(my_func, backend=compiler_fn)

    my_func_gm(torch.rand(3, 3), 5)

    print("done")


if __name__ == "__main__":
    test_fx_print()
