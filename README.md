# FX Tools

## Experiments with PyTorch FX

Main point of interest is probably the AD pass in https://github.com/awf/fx-tools/blob/main/DiffFX_Experiments.ipynb



- unlike shapeprop - don't actually run the calcullations
    - good for memory/time
    - good because shnty_propagator explicitly knows if it's OK to propagate

Good that we can get type, e.g. to resolve call_method

## Shape propagation

This is an aside -- we need shapes for e.g. the gradient of 'trace', so let's
quickly assemble a solution, where we can call `fx_shape` on proxies.

Quick hack here, as we expect more thorough handling upstream:
 - https://discuss.pytorch.org/t/fx-proxy-and-shapes/113861/4
 - https://github.com/pytorch/pytorch/issues/54982
 - https://www.youtube.com/watch?v=pLni96jtcjY

