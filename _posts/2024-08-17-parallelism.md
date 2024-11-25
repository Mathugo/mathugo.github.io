---
layout: single
classes: wide
author_profile: true
title: Short notes on types of parallelism for training neural networks
seo_title: A blog post on types of parallelism for training neural networks (distributed data parallel, model parallel, fdsp).


tags:
    - theory

published: true
---


As neural networks grow larger (see LLMs, though now it looks like we also have a trend towards smaller models with [Gemma2-2b](https://huggingface.co/google/gemma-2-2b) ) and datasets become more massive, parallelism techniques are crucial for efficient training. 
This is a short, far-from-exahustive list of different types of parallelism that can be found out there in the wild.

Obviously, all these methods assume you have multiple GPUs at your disposal (surprise!).

### 1. Data Parallelism

> TLDR: Split your dataset across multiple GPUs, each with a full model copy. Synchronize gradients after each pass.

Data parallelism is simple to implement, and scales well with number of devices for smaller model. It is especially effective for large datasets. On the other hand, it introduces a lot of communication overhead for gradient synchronization. Additionally, because a full copy of the model is stored on each device, it also causes memory redundancy.

Pseudocode:
```python
# On each device
for batch in dataloader:
    outputs = model(batch)
    loss = criterion(outputs, targets)
    loss.backward()
    
    # Synchronize gradients across devices
    all_reduce(model.parameters.grad)
    
    optimizer.step()
```

### 2. Model Parallelism

> TLDR: Divide your model across devices, each processes the same input at different stages.

Model parallelism is perfect for handling models too large for a single device. In doing so, it also reduces the memory required for a single device. Unfortunately, it might be complex to implement efficiently, because of potential load imbalance: it usually needs pipelining to avoid GPUs from remaining idle.

Pseudocode:
```python
# Define model portions
model_part1 = nn.Sequential(layer1, layer2).to('cuda:0')
model_part2 = nn.Sequential(layer3, layer4).to('cuda:1')

# Forward pass
def forward(x):
    x = model_part1(x)
    x = x.to('cuda:1')
    return model_part2(x)
```

### 3. Pipeline Parallelism

>TLDR: Split your model into stages on different devices. Data flows through the pipeline, with multiple batches processed simultaneously.

This is the solution to the imbalancing problem for plain model parallel. A nice explanation of model parallel + pipeline parallel can be found [here](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html). Pipeline parallel balances computation and communication and makes model parallelism more efficient. As a drawback, it requires a potentially complex scheduling: you have to deal with splitting the input across GPUs and schedule the pipeline. If you do this the wrong way, you might cause "bubble" periods of idle time.

Pseudocode:
```python
# Define stages
stage1 = nn.Sequential(layer1, layer2).to('cuda:0')
stage2 = nn.Sequential(layer3, layer4).to('cuda:1')

# Pipeline forward
def pipeline_forward(batches):
    for i, batch in enumerate(batches):
        x = stage1(batch)
        x = x.to('cuda:1')
        if i > 0:
            yield stage2(prev_x)
        prev_x = x
    yield stage2(prev_x)
```

### 4. Tensor Parallelism

> TLDR: Partition individual tensors (weights, activations) across devices. Each computes a portion of tensor operations.

This is somewhat on another level of abstraction wrt to Data and Model parallel, as tensor can represent anything in a deep learning pipeline. In other words, tensor parallel includes model and data parallel. 


Pseudocode:
```python
# Simplified tensor parallel linear layer
class TPLinear(nn.Module):
    def __init__(self, in_features, out_features, n_devices):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features // n_devices, in_features))
        
    def forward(self, x):
        local_out = F.linear(x, self.weight)
        return all_gather(local_out)
```

### 5. ZeRO (Zero Redundancy Optimizer)

> TLDR: Shards model parameters, gradients, and optimizer states across devices.

ZeRO includes all types of parallelism. More specifically, it impleemnts three possible options:

ZeRO offers three progressive levels of memory optimization. Each level increases memory efficiency but also introduces more communication overhead. ZeRO-3 provides the highest memory efficiency but with the most communication. More specifically:

- ZeRO-1: Optimizer State Partitioning: 
  - Partitions optimizer states (e.g., momentum buffers) across GPUs
  - Each GPU only stores optimizer states for its portion of parameters
  - Model parameters and gradients are still replicated on all GPUs

- ZeRO-2: Gradient Partitioning
  - Includes all of ZeRO-1
  - Additionally partitions gradients across GPUs
  - Each GPU only computes and stores gradients for its parameter portion
  - Model parameters are still replicated on all GPUs

- ZeRO-3: Parameter Partitioning
  - Includes all of ZeRO-1 and ZeRO-2
  - Additionally partitions model parameters across GPUs
  - Each GPU only stores a portion of the model parameters
  - Requires gathering parameters during forward/backward passes


ZeRO offers the most flexibility by combining benefits of data and model parallelism. Obviously, it introduces increased communication overhead and its complexity increases with higher ZeRO levels. Implementation of ZeRO is typically used through libraries like DeepSpeed or PyTorch's FSDP.

---



#### References
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [HF on parallelism](https://huggingface.co/docs/transformers/v4.15.0/parallelism)