import torch


def fx_print(gm):
    def commajoin(vs):
        return ",".join(vs)

    name2ord = {}
    ord = 10

    args = [n.name for n in gm.graph.nodes if n.op == "placeholder"]
    name = torch.nn.Module._get_name(gm)
    print(f"def {name}({commajoin(args)}):")
    for n in gm.graph.nodes:
        assert n.name not in name2ord
        name2ord[n.name] = f"v{ord}"
        ord += 1
        target = n.target.__name__ if n.op == "call_function" else n.target

        def argstr(a):
            if isinstance(a, torch.fx.Node):
                return name2ord[a.name]
            return str(a)

        args = [argstr(a) for a in n.args]
        pr = lambda x: print("  " + x)
        comment = f" # {n.meta['shnty']}" if 'shnty' in n.meta else ''
        if n.op == "placeholder":
            pr(f"{argstr(n)} = {target}{comment}")
        elif n.op == "call_function":
            pr(f"{argstr(n)} = {target}({commajoin(args)}){comment}")
        elif n.op == "call_method":
            pr(
                f'{argstr(n)} = {argstr(args[0])}.{target}({",".join(args[1:])}){comment}'
            )
        elif n.op == "output":
            pr(f"return {argstr(args[0])}{comment}")
        else:
            pr(f'# unhandled {n.op} {argstr(n)} = {target}({",".join(args)}){comment}')


def test_fx_print():
    # Just a non-crash test
    # TODO: code round-trip test
    def aux(p, q):
        return torch.relu(1.234 * p * q).neg()

    def my_func(x, b):
        y = 2 * x
        for _ in range(2):  # Loops will be unrolled
            x = aux(x, y)  # Function calls will be inlined
        return torch.atan2(b, x)

    gm = torch.fx.symbolic_trace(my_func)

    fx_print(gm)


if __name__ == "__main__":
    test_fx_print()
