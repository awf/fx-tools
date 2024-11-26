from typing import List
import torch
from collections import defaultdict

from .fx_print import fx_print_to_file

counts = defaultdict(lambda: 0)


def compiler_savegraph(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor] = []
):
    name = gm._get_name()
    counts[name] += 1
    name = name + "_" + str(counts[name])

    filename = f"tmp/{name}.py"
    fx_print_to_file(gm, filename)

    return gm
