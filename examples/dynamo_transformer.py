import logging
from typing import List
import torch
import torch._dynamo as tdy

from functorch import make_functional_with_buffers, vmap, grad, make_fx

from torchtext.models import RobertaEncoderConf, RobertaModel, RobertaClassificationHead

from fx_print import fx_print_iter

def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph: - saving to tmp.py")
    with open("tmp.py", "w") as f:
      print("""import torch
aten = torch.ops.aten
inf = float('inf')
""",file=f)
      for line in fx_print_iter(gm, ignore={'val', 'nn_module_stack','creation_timestamp','stack_trace'}):
        print(line, file=f)

    return gm.forward

tdy.config.log_level = logging.INFO
tdy.config.verbose=True

print('roberta')
vocab_size=256
roberta_encoder_conf = RobertaEncoderConf(vocab_size=vocab_size)
classifier_head = RobertaClassificationHead(num_classes=2, input_dim=768)
classifier = RobertaModel(encoder_conf=roberta_encoder_conf, head=classifier_head)

n_batch = 7
input = torch.randint(vocab_size, (n_batch,13))
labels = torch.rand((n_batch,2))
f, params, buffers = make_functional_with_buffers(classifier)

print('compile')
if False:
  f = torch.compile(f, backend=custom_backend)

import torchopt
# params = tuple(p.to(device="meta") for p in params)
optimizer = torchopt.adamw(lr=1e-4, weight_decay=0.01)
opt_state = optimizer.init(params)

def compute_loss(params, buffers, input):
    preds = f(params, buffers, input)
    return torch.nn.functional.mse_loss(preds, labels)

def training_step(params, opt_state, buffers, input):
    grads = grad(compute_loss)(params, buffers, input)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    return torchopt.apply_updates(params, updates)
  
from fx_shnty import shnty_trace
foo = make_fx(training_step)(params, opt_state, buffers, input)

custom_backend(foo,2)
# from functorch.compile import aot_function
# f = aot_function(f, custom_backend)



print('run')
print(training_step(params, opt_state, buffers, input))
