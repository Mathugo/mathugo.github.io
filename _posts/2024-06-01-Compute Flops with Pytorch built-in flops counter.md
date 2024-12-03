---
layout: single
author_profile: true
title: Flops with Pytorch built-in flops counter
seo_title: A blog post to explain how to use native pytorch to compute flops

tags:
    - theory

published: true
---
It is becoming more and more common to use FLOPs (floating point operations) to measure the computational cost of deep learning models. For Pytorch users, unfortunately, it looks like there is no agreed upon method or library to do that. 

After using different github libraries (see references), I found out that Pytorch actually has a built-in function to count flops. 

### How to count flops for a Pytorch model

I leave here a code snippet that shows how to compute the flops for a pytorch model only with forward or with forward and backward pass. We just need to provide the model and the input shapes for the model (or an input batch).

```python
import torch
from torch.utils.flop_counter import FlopCounterMode

def get_flops(model, inp: Union[torch.Tensor, Tuple], with_backward=False):
    
    istrain = model.training
    model.eval()
    
    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    total_flops =  flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops
```

Say you want to use the snippet to compute the flops for a resnet18, the you would do something like the following. 


```python
from torchvision.models import resnet18

model = resnet18()

get_flops(model, (1, 3, 224, 224))
```
I leave the discussion about whther FLOPs are actually a good way of measuring efficiency to [another blog post](https://alessiodevoto.github.io/Efficiency-metrics-in-Machine-Learning/#macs-and-flops)

#### References
- [pytorch flops counter](https://github.com/pytorch/pytorch/blob/main/torch/utils/flop_counter.py)
- [flopth](https://pypi.org/project/flopth/)
- [ptflops](https://pypi.org/project/ptflops/0.1/)