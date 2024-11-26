import torch
from fxtools import fx_type, fn_name

method_to_function_dict = {
    (torch.Tensor, "neg"): torch.neg,
    (torch.Tensor, "sum"): torch.sum,
}


def method_to_function(mod: torch.fx.GraphModule):
    g = mod.graph
    for n in g.nodes:
        if n.op == "call_method":
            key = (fx_type(n.args[0]), n.target)
            if key not in method_to_function_dict:
                print(f"Can't find {fn_name(key)} in method_to_function_dict")
                continue
            translation = method_to_function_dict[key]
            # create IR to call new activate
            with g.inserting_after(n):
                new_n = g.call_function(translation, n.args)
                n.replace_all_uses_with(new_n)
                g.erase_node(n)
        else:
            print("doing nothing to", n)

    mod.recompile()
    return None  # in-place modification of the graph
