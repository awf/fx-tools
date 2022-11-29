import torch


def _commajoin(vs):
    return ",".join(vs)


def fx_print_node(node, gm=None, name2ord=None):
    def argstr(a):
        if name2ord and isinstance(a, torch.fx.Node):
            return name2ord[a.name]
        return str(a)

    target = node.target.__name__ if node.op == "call_function" else node.target
    args = [argstr(a) for a in node.args]
    comment = f" # {node.meta['shnty']}" if "shnty" in node.meta else ""

    if node.op == "output":
        return f"return {argstr(args[0])}{comment}"

    lhs = argstr(node)
    if node.op == "placeholder":
        return f"{lhs} = {target}{comment}"

    if node.op == "call_function":
        return f"{lhs} = {target}({_commajoin(args)}){comment}"

    if node.op == "call_method":
        return f'{lhs} = {argstr(args[0])}.{target}({",".join(args[1:])}){comment}'

    if node.op == "get_attr":
        assert len(args) == 0
        if gm:
            val = getattr(gm, target)
            valstr = str(val).replace("\n", "\\n")[:40]
        else:
            valstr = "pass gm for value"
        return f"{lhs} = {target} # {valstr}"

    return f'# unhandled {node.op} {argstr(node)} = {target}({",".join(args)}){comment}'


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
