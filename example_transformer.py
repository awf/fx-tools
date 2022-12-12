import torch
import torch.nn as nn
import torch.fx

import torchtext
from torchtext.models import RobertaClassificationHead
from torchtext.functional import to_tensor

from fx_shnty import (
    shnty_trace,
    abstractify,
    fx_get_abstract_value_or_value,
    AbstractTensor,
)
from fx_print import fx_print

import difffx as dfx

ab_input = AbstractTensor(torch.Size((2, 13, 1024)), torch.float32)
ab_weights = AbstractTensor(torch.Size((250002,1024)),torch.float32)
dfx.vjp(torch.nn.functional.embedding, (ab_input, ab_weights))


print("loading")
xlmr_large = torchtext.models.XLMR_LARGE_ENCODER
classifier_head = torchtext.models.RobertaClassificationHead(
    num_classes=7, input_dim=1024
)
print("get_model")
model = xlmr_large.get_model(head=classifier_head, load_weights=False)

# Put model into inference mode (reduces runtime even without BT - esp for GPU execution, required for Better Transformer)
model.eval()

# Define input transform
print("get_transform")
transform = xlmr_large.transform()


input_batch = ["Hello world", "How are you freddie, great to see you!"]

print("input")
model_input = to_tensor(transform(input_batch), padding_value=1)

print("run")
print(model(model_input))

if False:
    # "symbolically traced variables cannot be used as inputs to control flow"
    model_gm = torch.fx.symbolic_trace(model)


# TODO: get this monkey off our patch
if True:
    orig_isinstance = torch.jit._isinstance

    def shnty_isinstance(x, ty):
        return orig_isinstance(fx_get_abstract_value_or_value(x), ty)

    torch.jit._isinstance = shnty_isinstance
    # end monkey


# jt = torch.jit.trace(model, (model_input, None))

model_gm = shnty_trace(model, (abstractify(model_input), None))
model_gm.recompile()

fx_print(model_gm)

print(model_gm(model_input))

m2 = shnty_trace(
    model.encoder.transformer.token_embedding,
    (AbstractTensor(torch.Size((2, 13)), torch.int64),),
)
fx_print(m2)


dfx.vjp(model.encoder.transformer.token_embedding, (AbstractTensor(torch.Size((2, 13)), torch.int64), ))


exit(0)


print("slow path:")
print("==========")
with torch.autograd.profiler.profile(use_cuda=False) as prof:
    for i in range(1):
        output = model(model_input)
print(
    prof.key_averages(group_by_stack_n=5).table(
        sort_by="self_cpu_time_total", row_limit=5
    )
)

print("fast path:")
print("==========")
with torch.autograd.profiler.profile(use_cuda=False) as prof:
    with torch.no_grad():
        for i in range(ITERATIONS):
            output = model(model_input)
print(
    prof.key_averages(group_by_stack_n=5).table(
        sort_by="self_cpu_time_total", row_limit=5
    )
)
