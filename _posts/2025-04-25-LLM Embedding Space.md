---
layout: single
classes: wide
author_profile: true
title: Visualizing the Token Embedding Space of an LLM
seo_title: What the vocabulary of an LLM (llama, phi-4) looks like if visualized in a plot.

published: true
---

> **Disclaimer**: *unlike other posts in this blog that actually served some purpose, this is just a random idea I had and am implementing for fun. So if your question is "why should I want to visualize the vocabulary of an LLM?", I don't have an answer* 😄

---

I recently realized I have never visualized the embedding space of an LLM. And while it's perfectly reasonable to go through life without ever seeing the inside of an LLM's vocabulary, I personally find it satisfying to make abstract things more concrete — especially when they're hiding in 4096-dimensional space.

LLMs are trained on vast corpora of text. To process text, they first tokenize it — splitting it into units ( =tokens) — and then represent each token as a dense vector in a high-dimensional space. These vectors are parameters of the model, learned during training.

The size of the vocabulary (i.e., how many distinct tokens the model knows) is a hyperparameter. For example, LLaMA 2: \~32,000 tokens, LLaMA 3: \~128,000 tokens, LLaMA 4: up to 200,000 tokens and so on. The token embedding dimension, that is the number of elements of each vector representing a token, (e.g., 4096) is another hyperparameter.

So if we take a LLaMA 3 model with an embedding size of 4096 and a vocabulary of 128k tokens, we’re dealing with 128k points in a  4096-dimensional space — not exactly human-interpretable. But what if we reduce the dimensionality? We’ll lose information, sure, but it might still give us interesting insights. 

As usual, you can also <a href="https://colab.research.google.com/drive/1mxFgj9R8s9nJoJUEisQxVPH9LRT4Lm6R" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---

### 🔧 What We'll Do

* Use a small corpus (a slice of Wikipedia) to filter only the tokens that appear in it — otherwise, we’d be visualizing 128k points.
* Use PCA to project the 4096-dimensional token vectors into 3D space (so we can plot them).
* Plot token embeddings interactively with Plotly, and visualize how an LLM represents the tokens in its latent space!

We'll use the Hugging Face Transformers library to load a LLaMA 2 model, but you can replace it with any other model !

```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16).eval()
```

---


To avoid plotting all 128k tokens, we’ll just extract those that actually appear in some real text. (If you want to visualize the **entire** vocabulary, I'll leave the code at the end of the post, but consider that might generate a rather large and unexplorable plot.)

```python
from datasets import load_dataset

# Load part of the Wikitext-2 dataset (raw text)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:20%]")
corpus = " ".join(dataset["text"])
```

---

We tokenize the corpus and count how often each token appears. This helps us later if we want to color or size points by frequency.

```python
from collections import Counter

# Tokenize the corpus
input_ids = tokenizer(corpus, return_tensors="pt")["input_ids"].flatten()
unique_ids = sorted(set(input_ids.tolist()))
freqs = Counter(input_ids.tolist())
```

---

Let’s extract the embeddings from the model and use PCA to reduce their dimension. This means that each token embedding will go from 4096 -> 3 dimensions. 

```python
from sklearn.decomposition import PCA

# Extract embeddings for tokens in corpus
embedding_weights = model.embed_tokens(torch.tensor(unique_ids).unsqueeze(0).to(model.device)).squeeze(0)

# Reduce to 3D using PCA
reduce_dims = PCA(n_components=3)
embeddings_3d = reduce_dims.fit_transform(embedding_weights.cpu().detach().float())
```

---

We tokenize the corpus and count how often each token appears. This helps us later if we want to color or size points by frequency.

```python
# Decode each token to its string representation
tokens = [tokenizer.decode([i]) for i in unique_ids]
sizes = [freqs[i] for i in unique_ids]
```

---

Now let’s put it all together in an interactive 3D plot. I'm clipping the tokens that appear more than 100 times because they are outliers that would cause all the other points to have the same color.

```python
import plotly.express as px

fig = px.scatter_3d(
    x=embeddings_3d[:, 0],
    y=embeddings_3d[:, 1],
    z=embeddings_3d[:, 2],
    hover_name=tokens,
    color={k: v if v < 100 else 100 for k, v in freqs.items()},  # clip outlier freqs
    title="Token Embeddings (Filtered by Corpus)",
)

fig.update_traces(marker=dict(size=3, showscale=False))

fig.write_html("my_token_embeddings.html") # save it to visualize locally
fig.show() 
```

And this is the results:

<iframe src="{{ site.url }}{{ site.baseurl }}/assets/html/llama7b.html" width="100%" height="600px" frameborder="0"></iframe>

It's really interesting to surf the plot and explore where learned to position different tokens in the embedding space! For example, on the top right corner there are a lot of words at the top (high `z` coordinate) seem to be names of cities or states: `Switzerland`, `Connecticut`, `irmingham` and so on ...  

---

## Extra 1: Visualizing all tokens in Dante's Divine Comedy

```python
# download the corpus from https://dmf.unicatt.it/~della/pythoncourse18/commedia.txt
import requests

url = "https://dmf.unicatt.it/~della/pythoncourse18/commedia.txt"
response = requests.get(url)
with open("commedia.txt", "w") as f:
    f.write(response.text)

dataset = load_dataset("text", data_files={"train": "commedia.txt"})
corpus = " ".join(dataset["train"]["text"])
```


## Extra 2: (Optional) Visualize the Entire Vocabulary

Try plotting the whole vocab, this will generate a large html plot.

```python
Uncomment at your own risk — slow and memory intensive!

full_ids = list(range(tokenizer.vocab_size))
full_embedding_weights = model.embed_tokens(torch.tensor(full_ids).unsqueeze(0).to(model.device)).squeeze(0)
full_embeddings_3d = PCA(n_components=3).fit_transform(full_embedding_weights.cpu().detach().float())
full_tokens = [tokenizer.decode([i]) for i in full_ids]

fig_full = px.scatter_3d(
    x=full_embeddings_3d[:, 0],
    y=full_embeddings_3d[:, 1],
    z=full_embeddings_3d[:, 2],
    hover_name=full_tokens,
    title="Token Embeddings (Full Vocabulary)"
)

fig_full.show()
```

---

## 🧠 Final Thoughts

This was just a fun exercise to make LLM internals a bit more tangible. Sure, PCA compresses a lot of high-dimensional nuance into 3D, but it's still fascinating to see structure emerge. Try zooming into different clusters, or color by other features (length, part-of-speech, etc.).



