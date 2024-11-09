---
layout: single
classes: wide
author_profile: true
title: Vision Transformer in *pure* JAX.
seo_title: Implementation of a Vision Transformers in pure JAX, no other frameworks/libraries.

published: true
---

I decided to do this for two reasons. The first reason is that, for years, I had to bear my Ph.D. advisor coming into the lab while I was happily coding my Pytorch model, slowly sneaking at my back, stare at my screen and say - with a disappointed look - "you should definitely do this in JAX". The second reason is this nice [blog post](https://neel04.github.io/my-website/blog/pytorch_rant/) from Neel Gupta.

<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

However, every time I tried to use JAX, I ended up using Flax instead, which offers a kind of object oriented interface (similar to torch). While Flax is great, it introduces additional layers of abstraction that make it similar to Pytorch and therefore I ended up wondering: "why am I doing this?". There are other great frameworks as well, with different functionalities, like equinox (maybe closer to JAX's original nature), but they always add "another layer".

This time, I wanted to take a taste of **bare JAX** and avoid external libraries or abstractions. In this implementation, I’ve built a basic Vision Transformer entirely from scratch. Although it may not be the most efficient code, my focus is to explore JAX directly and train a small model while leveraging JAX’s core features, like `vmap` and `jit`, without any external frameworks.

I will cover the following topics: 

1. Initialization of the weights (in pure JAX it can take a while)
2. Coding the ViT logic and parallelization (with `jax.vmap`)
3. Training with just in time (with `jax.jit`)

✋ If you are not interested in model initialization, you can just skip to the core part where we implement the [model and train it](https://alessiodevoto.github.io/ViT-in-pure-JAX/#the-model-is-just-a-function).

You can also <a href="https://colab.research.google.com/drive/1wBA1UUde72yMDvZ7ITS8cFAx90HDwD5D#scrollTo=SUBw2ZtVN7Lr" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

###  Vision Transfomer 
In the following, I assume you are already familiar with the Vision Transformer architecture. If you are not, you can take a look [here](https://arxiv.org/abs/2010.11929). In short, ViTs split images into patches, and treat patches as tokens (like words in NLP models), processing them using transformer layers with bidirectional (non masked) attention. In this post, we’ll build a small ViT that can train on the Imagenette dataset, and you can even run it on your local machine.


Speaking of GPUs, JAX offers seamless handling of hardware acceleration. It automatically detects and utilizes available GPUs/TPUs without requiring explicit code changes.


```python
import jax

print("Available devices:", jax.devices()) # JAX will take care of the device placement for you
```

    Available devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), TpuDevice(id=1, process_index=0, coords=(0,0,0), core_on_chip=1), TpuDevice(id=2, process_index=0, coords=(1,0,0), core_on_chip=0), TpuDevice(id=3, process_index=0, coords=(1,0,0), core_on_chip=1), TpuDevice(id=4, process_index=0, coords=(0,1,0), core_on_chip=0), TpuDevice(id=5, process_index=0, coords=(0,1,0), core_on_chip=1), TpuDevice(id=6, process_index=0, coords=(1,1,0), core_on_chip=0), TpuDevice(id=7, process_index=0, coords=(1,1,0), core_on_chip=1)]


In this notebook, we are going to use a small ViT, with the following hyperparameters:

```python
image_size = 64
patch_size = 4
num_patches = (image_size // patch_size) ** 2

num_layers = 4      # number of transfomer layers
hidden_dim = 192    # hidden dimension of each token
mlp_dim = 192*4     # hidden dimension in the MLP 

num_classes = 10    # Imagenette number of classes
num_heads = 4       # attention heads
head_dim = hidden_dim//num_heads
```

### Initializing the model

JAX is a fully functional framework, which means that model parameters are treated as a distinct set of numbers, existing "outside" the model itself. This gives you a nice, low-level feel for how the model works. Instead of encapsulating parameters within an object (like in torch), you’re directly manipulating a concrete set of weights along with a function that processes them.

To initialize these weights at random, we need some random primitives (just like in torch). In JAX, every call to a random primitive requires a random key, which ensures that the randomness is both explicit and controllable.  This means that instead of going 

```python
a_tensor = torch.randn(tensor_shape)
```

you have to explicitly allocate a key first and then use it to generate a random number, like this:

```python
key = random.PRNGKey(42)
a_tensor = random.normal(key, tensor_shape)
```

This is great for ML practitioners, and you know what I'm talking about if you ever had to use [torch random seeding and ended up with reproducibility issues](https://neel04.github.io/my-website/blog/pytorch_rant/#seeding). The main reason for JAX explicitly tracking the random keys without using a global random state is that this would compromise the execution of parallel code, that is one of the main perks of JAX. You can read more about randomness in JAX [here](https://jax.readthedocs.io/en/latest/jax.random.html).

Let's see what parameters we need for our ViT. Here is the list:

- a `CLS` (classification) token
- a projection to transform the patches into tokens
- a positional encoding
- N transformer blocks made of multihead attention and Feed Forward MLP
- a final head for classification


As mentioned before, these parameters are just numbers that we can store in a dictionary like this:


```python
# initialize vit parameters
vit_parameters = {
    'patch_embed': None,
    'positional_encoding': None,
    'layers': [],
    'final_layer_norm': None,
    'head': [],
    'cls_token': None
}
```


We now need to initialize each set of weights separately. Again, we could use a library for this, like optax, but we want to go through the process manually to better understand what's happening under the hood.

We initialize the class token with all zeros:
```python
import jax.numpy as jnp # this is what we use to manipulate tensors in JAX. It is supers similar to numpy

# for the class token, we just need a single vector of the same size as a token
cls_token = jnp.zeros((1,hidden_dim))
vit_parameters['cls_token'] = cls_token
```

For patch embedding, positional encoding, final head, and all transformer blocks we use random values (check the colab for complete code).
Each transformer block is made up of `attention`, `mlp` and `layer normalization`. We define a function to initialize each of these components. I'll show just the mlp initialization here for brevity. I'll do it using [Xavier intialization](https://paperswithcode.com/method/xavier-initialization), but this is not crucial and you can just use a random normal.

For the MLP, we need weights and biases for 2 layers.
```python
def initialize_mlp(hidden_dim, mlp_dim, key):
    w1_key, w2_key = random.split(key) # get new random keys from the one provided

    # Xavier uniform limit for w1 and w2
    limit = jnp.sqrt(6.0 / (hidden_dim + mlp_dim))

    # Xavier uniform initialization for weights
    w1 = random.uniform(w1_key, (hidden_dim, mlp_dim), minval=-limit, maxval=limit)
    b1 = jnp.zeros(mlp_dim)

    w2 = random.uniform(w2_key, (mlp_dim, hidden_dim), minval=-limit, maxval=limit)
    b2 = jnp.zeros(hidden_dim)

    return w1, b1, w2, b2
```

We are now ready to initialize all the weights in each transformer layer! Let's create a set of parameters for each layer and store them in our dictionary.
```python
key = random.PRNGKey(42)
key, *layer_keys = random.split(key, num_layers+1)

for i in range(num_layers):
    mlp_params = initialize_mlp(hidden_dim, mlp_dim, layer_keys[i])
    attn_params = initialize_attention(hidden_dim, num_heads, layer_keys[i])
    ln1_params = initialize_layer_norm(hidden_dim)
    ln2_params = initialize_layer_norm(hidden_dim)
    vit_parameters['layers'].append((mlp_params, attn_params, ln1_params, ln2_params))

# we also have a final layer norm outside the loop
final_layer_norm_params = initialize_layer_norm(hidden_dim)
vit_parameters['final_layer_norm'] = final_layer_norm_params
```

Finally, we can now write the code for the transformer encoder!


### The Model is Just a Function

One thing we quickly notice about JAX is that everything is a function — including models. This is very different from torch, where we usually look at the model as a composition of objects (`nn.Module`s). So we’ll write the forward pass as  *just* a function. The ViT function will take the ViT parameters and an image as input, that is:

```python
prediction = vit_function(image, vit_parameters)
```

As we can see, the parameters are "outside" of the model. Before writing the actual code for the ViT, we come to another special feature o JAX: *parallelization*. Thanks to JAX native `vmap` function, we'll just pretend there is no batch dimension and then use `vmap` to automagically handle batches. This is a great improvement as we don't have to reason in one additional dimension and there will be no need for stuff like `batch,sequence,dim = input.shape` (unlike torch.) So from now on, we'll just ignore the batch dimension.

Don't forget that each transformer block is nothing but a *function* over the **model parameters** and an **input**.
For the MLP, we just perform an up and down projection with a Relu activation function in the middle. Notice that we will get the input parameters from the dictionary we created earlier.
```python
def mlp(x, mlp_params):

    # unpack the parameters
    w1, b1, w2, b2 = mlp_params

    # out = (Relu(x*w1 + b1))*w2 + b2
    up_proj = relu(jnp.matmul(x, w1) + b1)
    down_proj = jnp.matmul(up_proj, w2) + b2

    return down_proj
```

Now self attention, the only catch here is to project into multiple heads and then concatenate back
```python
def self_attention(x, attn_params):
    # unpack the parameters
    q_w, k_w, v_w, q_b, k_b, v_b = attn_params
    n, d_k = x.shape   # n and d_k are the sequence length of the input and the hidden dimension

    # project the input into the query, key and value spaces
    q = jnp.matmul(x, q_w) + q_b
    k = jnp.matmul(x, k_w) + k_b
    v = jnp.matmul(x, v_w) + v_b

    # reshape to have heads
    # n, (num_heads head_dim) ->  (n, num_heads, headim) -> (num_heads, n, head_dim)
    q = q.reshape(n, num_heads, head_dim).swapaxes(0, 1)
    k = k.reshape(n, num_heads, head_dim).swapaxes(0, 1)
    v = v.reshape(n, num_heads, head_dim).swapaxes(0, 1)


    # perform multi-head attention
    attention_weights_heads = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / jnp.sqrt(head_dim)
    attention_weights_heads = jax.nn.softmax(attention_weights_heads, axis=-1)

    # output projection
    output = jnp.matmul(attention_weights_heads, v)
    output = output.swapaxes(0,1).reshape(n, d_k)
    return output
```

Finally, we can assemble attention and mlps into a transformer block.
```python
def transformer_block(inp, block_params):

    # unpack the parameters
    mlp_params, attn_params, ln1_params, ln2_params = block_params

    # attention
    x = layer_norm(inp, ln1_params)
    x = self_attention(x, attn_params)
    res = x + inp # skip connection

    # mlp
    x = layer_norm(res, ln2_params)
    x = mlp(x, mlp_params)
    x = x + res
    return x
```

Before feeding the first image to the model, we need one more additional step to transform an input image into a sequence of patches. To do that, we use [einops](https://einops.rocks/), which offers a highly expressive interface to reshape tensors. Another way would be applying convolutions but here we are just using bare JAX code so we get a sequence of tokens from an image like this:

```python
from einops import rearrange
patches = rearrange (image, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
```
With this, we are ready to go! The final transformer then works by:
1. reshaping the image into patches
2. projecting patches into tokens
3. adding a class token and positional embeddings
4. looping through a stack of transformer blocks
5. applying the final classification head

Let's implement these steps:

```python
def transformer(image, vit_parameters):
    # reshape image from c,h,w -> num_patches, patch_size*patch_size
    patches = rearrange (image, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

    # embed the patches into tokens
    patches = jnp.matmul(patches, vit_parameters['patch_embed'])

    # add positional encoding
    patches = patches + vit_parameters['positional_encoding']

    # append class token to sequence
    cls_token = vit_parameters['cls_token']
    patches = jnp.concatenate([cls_token, patches], axis=0)

    # forward through all transformer blocks
    for layer, block_params in enumerate(vit_parameters['layers']):
        patches = transformer_block(patches, block_params)

    # final layer norm
    patches = layer_norm(patches, vit_parameters['final_layer_norm'])

    # get the class token and apply the final head
    patches = patches[0, :]
    logits = jnp.matmul(patches, vit_parameters['head'][0]) + vit_parameters['head'][1]
    return logits
```

Let's test it on a random input:

```python
sample_image = random.normal(key, (3 ,image_size, image_size))
prediction = transformer(sample_image, vit_parameters)
print("Output shape:", prediction.shape) # should be (num_classes,)
```

    Output shape: (10,)

As you may have noticed, the random input is just an image without a batch dimension. Let's see how we can add a batch dimension without modifying the code.

### Vectorized Mapping with `vmap`

As anticipated, before jumping into training, we'll look at one of the coolest features JAX offers: `vmap`. This allows you to vectorize your functions, meaning you can apply them over batches of data without writing explicit loops. In a way, it’s like automatic batching. You write a function that works on a single example, and `vmap` will apply it to all examples in a batch in one go.

For example, if you have a function that processes a single image, you can turn it into a function that processes an entire batch of images with just one line:

```python
batched_fn = jax.vmap(single_image_fn)
```

This can come in handy when applying the model over a batch of data. This means that we can run one pass of our transformer over a batch of images very easily. Let's try:

```python
bsize = 5
sample_images = random.normal(key, (bsize, 3 ,image_size, image_size))

# if we apply the transformer to a batch of images, we should get a batch of logits
# but this will raise an error
# prediction = transformer(sample_images, vit_parameters)
```

Let's apply `vmap`. We need to map each input in the batch (first dimension is 0) to *all* the parameters (second dimension in `None`).


```python
prediction = jax.vmap(transformer, in_axes=(0, None))(sample_images, vit_parameters)
print("Prediction shape:", prediction.shape)
```

    Prediction shape: (5, 10)


Actually, `vmap` can do way more than this. I recommend this [blog](https://jiayiwu.me/blog/2021/04/05/learning-about-jax-axes-in-vmap.html) for an overview.

### Loss Function

Next up is the loss function. We’ll use the Cross-Entropy Loss, which is a standard choice for classification tasks.


```python
def cross_entropy_loss(patches, vit_parameters, ground_truth):
    prediction = jax.vmap(transformer, in_axes=(0, None))(patches, vit_parameters)
    logs = jax.nn.log_softmax(prediction)
    l = -jnp.mean(jnp.sum(ground_truth * logs, axis=-1))
    return l
```


```python
l = cross_entropy_loss(sample_images, vit_parameters, jnp.zeros((bsize, 10)).at[0, 1].set(1))
print("Loss:", l)
```

    Loss: 0.49248534


### Dataset Loading (Stealing From PyTorch)

For dataset loading, I’m going to steal some code from PyTorch. PyTorch’s data utilities work really well, and since this isn’t a post about data loading, we’ll skip the hassle of reinventing the wheel here.


```python
from torchvision.datasets import CIFAR10, Imagenette
from torchvision import transforms
from torch.utils.data import DataLoader

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_dataset = Imagenette(
    root='/home/aledev/datasets/imagenette3',
    size="160px",
    split='train',
    download=False,
    transform=transforms.Compose([transforms.Resize((image_size,image_size)),  transforms.ToTensor(), transforms.Normalize(mean, std)])
    )
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

test_dataset = Imagenette(
    root='/home/aledev/datasets/imagenette3',
    size="160px",
    split='val',
    download=False,
    transform=transforms.Compose([transforms.Resize((image_size,image_size)), transforms.ToTensor(), transforms.Normalize(mean, std)])
    )
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
```

Let's code a simple evaluation function that loops over the test data and computes accuracy


```python
from tqdm import tqdm

def eval(vit_parameters):

  correct = 0

  for(img, target) in tqdm(test_loader, desc="Eval", unit="item"):

    img = jnp.asarray(img, dtype=jnp.float32)
    target = jnp.asarray(target)

    logits = jax.vmap(transformer, in_axes=(0, None))(img, vit_parameters)
    prediction = jnp.argmax(logits, axis=-1)
    correct += jnp.sum(prediction == target).item()

  acc = correct / len(test_dataset)

  return acc

accuracy = eval(vit_parameters)
print("Accuracy before training", accuracy)
```

    Eval: 100%|██████████| 16/16 [00:18<00:00,  1.13s/item]

    Accuracy before training 0.06394904458598726


    


### Training and Just in Time compilation

Before we dive into training, we meet another cool feature of JAX: `jit`, that is, just in time compilation. One of JAX’s biggest selling points is its ability to automatically compile and optimize your code using just-in-time (JIT) compilation. With JAX, you can wrap your functions in `jax.jit()` to make them faster by turning them into optimized code. It’s a one-liner and can massively speed up your training loop. In essence, `jit` lets you write Python code, and JAX will magically optimize it behind the scenes. It’s not even hard to use, so there’s no reason *not* to take advantage of it!

Here’s how you can JIT-compile your training step:

```python
@jax.jit
def train_step(params, data):
    # your training step logic here
```

#### Parameter updates

This is where JAX differs most from PyTorch. In PyTorch, you call `.backward()` on your loss, and it handles everything, i.e. computes loss and gradients, that you'll find stored in your model parameters. In JAX, you need to manually compute gradients and update parameters yourself, which gives you a more hands-on experience with the inner workings of optimization.

To perform gradient descent, we’ll compute the gradient of the loss with respect to the parameters. In JAX, you can do this using the `jax.values_and_grad` function:

```python
loss, grads = value_and_grad(cross_entropy_loss, argnums=1)(input, parameters, targets)
```

What will the gradients look like ? The gradients are just going to be a dictionary (pytree) with the same keys as the model parameters, but instead of holding the parameters, they will hold the gradients.


```python
from jax import value_and_grad

# fake labels and images
sample_images = random.normal(key, (bsize, 3 ,image_size, image_size))
sample_target = jnp.zeros((bsize, 10)).at[0, 1].set(1)
current_loss, grads = value_and_grad(cross_entropy_loss, argnums=1)(sample_images, vit_parameters, sample_target)

print("Current loss:", current_loss)
print("Gradients:", grads.keys())
```

    Current loss: 0.49248534
    Gradients: dict_keys(['cls_token', 'final_layer_norm', 'head', 'layers', 'patch_embed', 'positional_encoding'])


We now have a dictionary of gradients that mirrors the structure of our parameters. To update the parameters, we’ll perform simple gradient descent. The update rule is:

$$ \theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \nabla_\theta L(\theta) $$

Where:
- $$ \theta $$ is the parameter we’re updating, in our case the dictionary.
- $$ \eta $$  is the learning rate.
- $$ \nabla_\theta L(\theta) $$ is the gradient of the loss with respect to the parameter, in our case the gradients dictionary.

JAX has some great libraries for optimization, like `optax`, but for simplicity, we’ll just manually update the parameters using vanilla SGD. Notice that to do this we'd have to go throught the dictionary and update all values that have the same key. Fortunately, JAX has a function that does that for us: `jax.tree.map`.

We just have to tell the gradient descent rule:

```
updated_params = jax.tree.map(lambda p, g: p - 0.01 * g, vit_parameters, grads)
```

Putting everything together, the training step will look like this:


```python
from jax import jit, value_and_grad

@jit
def train_step(patches, vit_parameters, target_one_hot):
    # compute gradients
    current_loss, grads = value_and_grad(cross_entropy_loss, argnums=1)(patches, vit_parameters, target_one_hot)

    # update parameters
    updated_params = jax.tree.map(lambda p, g: p - 0.01 * g, vit_parameters, grads)

    return current_loss, updated_params
```

Finally, let's train the model. We don't expect any special results because we training without any optimization and with a super small model. Also, I'll only train for 30 epochs here, but you can let it go on for longer.


```python
import jax
from jax import value_and_grad
from jax import jit
from tqdm import tqdm

num_epochs = 20


for epoch in range(num_epochs):

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
    #for (data, target) in tqdm(train_loader, desc=f'Train epoch {epoch}'):
    for i, (data, target) in progress_bar:

        # convert to numpy
        data = jnp.asarray(data)
        target = jnp.asarray(target)

        # reshape and get one hot fot loss
        target_one_hot = jax.nn.one_hot(target, num_classes)

        current_loss, vit_parameters = train_step(data, vit_parameters, target_one_hot)

        progress_bar.set_postfix({'loss': current_loss})

    eval_acc = eval(vit_parameters)
    print(f'Epoch: {epoch}, Eval acc: {eval_acc}')
```

    Epoch 1/20: 100%|██████████| 37/37 [00:29<00:00,  1.26it/s, loss=2.1218076]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.17item/s]


    Epoch: 0, Eval acc: 0.2573248407643312


    Epoch 2/20: 100%|██████████| 37/37 [00:17<00:00,  2.07it/s, loss=2.162859]
    Eval: 100%|██████████| 16/16 [00:07<00:00,  2.10item/s]


    Epoch: 1, Eval acc: 0.25095541401273885
    ...


    
Hope you enjoyed this, please reach me at https://alessiodevoto.github.io/ if you have any questions or find inconsistencies!

Thanks to [Rahil](https://x.com/rahilpandya) for spotting a bug in the attention shapes!

Thanks [Luigi](https://luigisigillo.github.io/) and [Jary](https://jarypomponi.com/) for reviewing this!
