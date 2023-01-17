import logging
from typing import List
import torch
from collections  import defaultdict
import torch._dynamo as tdy

import torch.nn.functional as F
from torch.utils._pytree import tree_map

from functorch import make_functional_with_buffers, vmap, grad, make_fx, functionalize

from fx_print import fx_print_to_file

from savegraph import compiler_savegraph

import pdb


tdy.config.log_level = logging.INFO
tdy.config.verbose=True


n_batch = 7
labels = torch.randint(10, (n_batch,),  requires_grad=False)
inputs = torch.rand((n_batch,1,28,28),  requires_grad=False)

class Net(torch.nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
      self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
      # tiny dropouts just to get some repeatability, but not optimize away
      self.dropout1 = torch.nn.Dropout(0.000001) 
      self.dropout2 = torch.nn.Dropout(0.000001)
      self.fc1 = torch.nn.Linear(9216, 128)
      self.fc2 = torch.nn.Linear(128, 10)

    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # Run max pooling over x
      x = F.max_pool2d(x, 2)
      # Pass data through dropout1
      x = self.dropout1(x)
      # Flatten x with start_dim=1
      x = torch.flatten(x, 1)
      # Pass data through fc1
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output

classifier = Net()
classifier_func, params, buffers = make_functional_with_buffers(classifier)

print(classifier(inputs).T[0], classifier_func(params, buffers, inputs).T[0])
print(classifier(inputs).T[0] - classifier_func(params, buffers, inputs).T[0])

def tree_norm(xs):
    return "T<"+",".join(tree_map(lambda x: f"{torch.norm(x).item():.3g}", xs))+">"

import torchopt

classifier_func, params, buffers = make_functional_with_buffers(Net())
optimizer = torchopt.adamw(lr=1e-4, weight_decay=0.01)
opt_state = optimizer.init(params)


@torch.no_grad()
def compute_loss(params, buffers, inputs, labels):
    preds = classifier_func(params, buffers, inputs)
    return torch.nn.functional.nll_loss(preds, labels)

@torch.no_grad()
def training_step(params, opt_state, buffers, inputs, labels):
    loss = compute_loss(params, buffers, inputs, labels)
    grads = grad(compute_loss)(params, buffers, inputs, labels)
    updates, new_opt_state = optimizer.update(grads, opt_state, params=params)
    new_params = torchopt.apply_updates(params, updates)
    return new_params, new_opt_state, loss

for iter in range(8):
  params, opt_state, loss = training_step(params, opt_state, buffers, inputs, labels)
  print(loss, tree_norm(params))

if False:
  # # torch._dynamo.exc.InternalTorchDynamoError on 2.0.0a0+git5adc18d

  # gm = torch._dynamo.export(classifier_func, (params, buffers, inputs), aten_graph=True)
  # fx_print_to_file(gm, 'tmp.py')

  # # Calls down to torch._dynamo.optimize
  cc = torch.compile(classifier_func, backend=compiler_savegraph, dynamic=True)
  cc(params, buffers, inputs)
  exit(0)
  foo = torch.compile(compute_loss, backend=compiler_savegraph)
  loss = foo(params, buffers, inputs, labels)

for p in params:
   p.requires_grad = False

print('LOSS', compute_loss(params, buffers, inputs, labels))
training_step(params, opt_state, buffers, inputs, labels) # check it runs...

# works...
foo = make_fx(functionalize(grad(compute_loss)))(params, buffers, inputs, labels)
compiler_savegraph(foo)

foo_graph = make_fx(functionalize(training_step))(params, opt_state, buffers, inputs, labels)

foo_compiled = compiler_savegraph(foo_graph)

# from functorch.compile import aot_function
# classifier_func = aot_function(training_step, compile)
# classifier_func(params, opt_state, buffers, inputs)

print('run')
for iter in range(8):
  params, opt_state, loss = foo_compiled(params, opt_state, buffers, inputs, labels)
  print(loss)
