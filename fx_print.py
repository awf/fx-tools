import torch


def _commajoin(vs):
    return ",".join(vs)


def fn_name(f):
    n = f.__name__
    if hasattr(f, "__module__") and f.__module__ != "_operator":
        n = f"{f.__module__}.{n}"

    if hasattr(f, "__code__"):
        n += f"[{f.__code__.co_filename}:{f.__code__.co_firstlineno}]"

    return n


def fx_print_node(node, gm=None, name2ord=None):
    def argstr(a):
        if isinstance(a, tuple):
            return "(" + _commajoin(map(argstr, a)) + ")"
        if isinstance(a, torch.fx.Node):
            return name2ord[a.name] if name2ord else a.name
        if isinstance(a, (int, float)):
            return f"{type(a).__name__}({a})"
        if isinstance(a, str):
            return f"'{a}'"

        return str(a) + f"[{type(a)}]"

    argstrs = [argstr(a) for a in node.args]
    comment = f" # {','.join(str(k) + ':' + str(v) for k,v in node.meta.items())}"

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
                valstr = str(val).replace("\n", "\\n")[:40]
            else:
                valstr = f"no attr {node.target}"
        else:
            valstr = "[no GM]"
        return f"{lhs} = getattr({node.target}, {node.args}) # {valstr}"

    return (
        f"# unhandled {node.op} {lhs} = {node.target}({_commajoin(argstrs)}){comment}"
    )


def fx_print(gm):

    name2ord = {}
    ord = 10

    args = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
    name = torch.nn.Module._get_name(gm)
    print(f"def {name}({_commajoin(args)}):")
    for node in gm.graph.nodes:
        assert node.name not in name2ord
        name2ord[node.name] = f"v{ord}"
        ord += 1

        node_str = fx_print_node(node, gm, name2ord)
        print("  " + node_str)


def test_fx_print():
    # Just a non-crash test
    # TODO: code round-trip test
    def aux(p, q):
        return torch.relu(1.234 * p * q).neg()

    def my_func(x, b):
        y = 2 * x
        for _ in range(2):  # Loops will be unrolled
            x = aux(x, y)  # Function calls will be inlined
        x = x.neg()
        x = x @ torch.eye(3)
        return torch.atan2(b, x)

    gm = torch.fx.symbolic_trace(my_func)

    fx_print(gm)


if __name__ == "__main__":
    test_fx_print()
