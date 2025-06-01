---
layout: single
classes: wide
author_profile: true
title: Peeking Inside Diffusion Language Models with LogitLens
seo_title: Interpretability on Large Diffusion Language Models with Logit Lens.

published: true
---


Lately, diffusion-based language models like [LLaDA](https://arxiv.org/abs/2502.09992) and [MMaDA](https://arxiv.org/abs/2505.15809) have been gaining traction. These aren't your standard left-to-right text generators - they’re bidirectional models trained to fill in missing tokens, more akin to BERT but on steroids. During training, Diffusion Language Models (DLMs) learn to predict `<mask>` tokens given context, effectively learning a denoising task.

In this post, we’ll explore how DLMs work "under the hood" using one of the most elegant interpretability tools out there: **LogitLens**. The final result will be a plot like this one, that we obtained applying LogitLens to Llama in a previous blog post.

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/logitlens/logitlens_small.png" alt="Llama Logitlens" style="max-width: 100%; width: 90%; display: block; margin: 0 auto;">

As usual, here is the code to reproduce all the plots, just <a href="https://colab.research.google.com/drive/1XfDN_3w4W6Us8mvDQRkehZVdfLJW82VQ" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## A Quick Refresher: What Is LogitLens?

[LogitLens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) is a clever interpretability method originally designed for autoregressive (AR) LLMs. Instead of applying the final language modeling head only to the last hidden layer (as is standard), LogitLens applies it at every intermediate layer. This lets us "see" how predictions evolve across layers—like watching a thought form in real time.

In a previous blog, I showed how to implement LogitLens for LLMs from scratch. This time, we’ll use it to analyze DLMs—specifically LLaDA and Dream—to observe how they construct predictions layer by layer. I should mention here that this is not the most advanced interpretability method (e.g. see this [nice blog](https://www.soniajoseph.ai/the-logit-lens-can-be-deceptive-if-not-used-properly/)) but it still gives us cool insights into the model's inner workings! 

---

## Setup: Probing LLaDA with a Simple Prompt

Let’s start by loading the LLaDA-8B model and giving it a classic prompt:

```python
import torch
from transformers import AutoTokenizer, AutoModel

device = 'cuda'
model_name = 'GSAI-ML/LLaDA-8B-Base'
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
```

Now we create a simple prompt:

```python
example = "The quick brown fox jumps"
prompt = tokenizer(example, return_tensors="pt").to(device).input_ids
```

Now, unlike like GPT-style models that generate one token at a time, diffusion models need to know their target length upfront. It's a bit like being asked to fill in a crossword puzzle, you need to know how many squares you're working with *in advance*. This might sound like a desing problem for open ended generation (it kind of is) but we can circumvent with [various strategies](https://arxiv.org/abs/2503.09573).

To prepare our input, we create a sequence with the original prompt + 8 mask tokens `<mask>`. The `<mask>` tokens are the empty squares of the puzzle the model has to fill in.


```python
mask_id=126336 # specific to llada model
gen_length=8
x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
x[:, :prompt.shape[1]] = prompt.clone()
```
Just to make sure we see what we are doing, let's visualize what is happening and print the input before feeding it to the model:

```python
tokenizer.convert_ids_to_tokens(x[0], skip_special_tokens=False)
```
    ['The', 'quick', 'brown', 'fox', 'jumps', '<|mdmmask|>', '<|mdmmask|>', '<|mdmmask|>', '<|mdmmask|>', '<|mdmmask|>', '<|mdmmask|>', '<|mdmmask|>', '<|mdmmask|>']

It makes sense, we have our prompt + the mask tokens the model will try to fill in. 

## Running the Model
Let’s forward the input through the model and grab the predicted tokens:

```python
# we need all the intermediate hidden states
with torch.no_grad():
    outputs = model(x, output_hidden_states=True)

# get the predicted token ids
predicted_token_ids = outputs["logits"].argmax(-1)

# convert the token ids to tokens
tokenizer.convert_ids_to_tokens(predicted_token_ids[0], skip_special_tokens=False)
```

    ['_', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.', '_', '_', 'jumps']

Well, after just one decoding step we were not expecting any special results. Usually, we perform at least N/2 denoising steps, where N is the generated sequence length.However, we see the model has already guessed (correclty) almost all the missing words, but the rest is kind of messy. If you want to get a better prediction, you should perform additional diffusion steps, by remasking some of the input tokens. This is out of the scope for this blog, as we just want to explore the inner representations of the model.

## Logit Lens
We just got the predictions fot the *final* output of the model, that is, what comes out of the last layer. What if we apply the model head (our "Lens") to intermediate representations though? Intermediate representations are stored in the `outputs.hidden_states`. We'll have one for each layer of the model and each token of the sequence.

```python
logitlens = []

for hidden_state in hidden_states:
    hidden_state = model.model.transformer.ln_f(hidden_state)
    logits = model.model.transformer.ff_out(hidden_state)

    predicted_token_ids = logits.argmax(-1)
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids[0], skip_special_tokens=False)

    logitlens.append(predicted_tokens)
```

Now the list contains decoded sequences for each layer. For instance, if we want to check what the model was "thinking" at layer 5, we can just do:

```python
print(logitlens[5])
```
    ['paralleled', 'awaited', 'funnels', 'gfx', 'tesy', 'awaited', 'awaited', 'blockList', 'blockList', 'blockList', 'blockList', 'blockList', 'blockList']


The raw output appears chaotic and difficult to interpret. To better understand the model's internal reasoning process, let's visualize the complete sequence predictions across all transformer layers using a heatmap (full implementation available in the accompanying notebook).

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/logitlensdiff/output_llada.png" alt="Llada Logitlens" style="max-width: 100%; width: 90%; display: block; margin: 0 auto;">

The bottom row shows the original input tokens, while each ascending row represents successive transformer layers from early to final. The color intensity indicates prediction confidence at each layer, with darker cells representing higher entropy (model uncertainty) and brighter cells showing lower entropy (confident predictions).
 --  that is why brighter cells are only in the final layers.

The most evident pattern is that coherent, confident predictions only emerge in the final layers, typically around layer 25 and beyond. Throughout the majority of the network depth, the model remains highly uncertain about its predictions, as evidenced by the prevalence of darker cells (and wrong tokens) in earlier layers. This behavior fundamentally differs from autoregressive models like LLaMA or Gemma, where confident predictions typically crystallize much earlier in the network.

Don't forget that we're observing only the first denoising step of what is inherently a multi-step iterative process. The model is designed to gradually refine predictions across multiple iterations, so initial uncertainty is not only expected but necessary for the denoising mechanism to function properly. Also, when compared to LLama's functioning, the complexity increases significantly because unlike autoregressive models that predict one token aftern a sequentially, diffusion models attempt to predict **all tokens simultaneously**. This parallel prediction task is inherently more challenging and requires deeper processing to resolve the interdependencies between different positions in the sequence.

Let's see what happens if we provide a sequence with a single mask token (just wondering whether the model will be more confident about its prediction when only one guess has to be made.)

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/logitlensdiff/output_llada2.png" alt="Llada Logitlens" style="max-width: 100%; width: 90%; display: block; margin: 0 auto;">

It looks like also in this case the final output takes shape at the very final layers!  Let’s repeat the process with Dream (check the Colab for the full code). Here, I will plot fewer layers to make it more clear:

<img src="{{ site.url }}{{ site.baseurl }}/assets/images/logitlensdiff/output_dream.png" alt="Dream Logitlens" style="max-width: 100%; width: 90%; display: block; margin: 0 auto;">

It's worth to point out the interesting behavior of Dream -- that we didn't observe in the LLada example. It's clear that sometimes Dream can "peek into the future." For instance, it predict tokens related to tokens that haven’t been revealed yet—like seeing the word "Guinea" emerge before "pig" (I'm assuming the sentence "this sentence is about a Guinea pig is not present in the training corpus here). This behavior would be impossible in autoregressive models, where each token only attends to previous ones. With DLMs, all tokens attend to each other from the start, enabling these models to make globally-informed guesses much earlier in the network! 

That's it! If you have feedback or find bugs, feel free to reach out on [Twitter](https://x.com/devoto_alessio) or any [social platform](https://alessiodevoto.github.io/). Thanks for reading!


