## The Original Transformer (PyTorch) :computer: = :rainbow:
This repo contains PyTorch implementation of the original transformer paper (:link: [Vaswani et al.](https://arxiv.org/pdf/1706.03762.pdf)). <br/>
It's aimed at making it **easy for beginners** to start playing and learning about transformers. <br/>

**Important note:** I'll be adding a jupyter notebook soon as well!

## Table of Contents
  * [What are transformers?](#what-are-transformers)
  * [Understanding transformers](#understanding-transformers)
  * [Machine translation](#machine-translation)
  * [Setup](#setup)
  * [HW requirements for training](#hw-requirements-for-training)

## What are transformers

Transformers were originally proposed by Vaswani et al. in a seminal paper called [Attention Is All You Need](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf).

You probably heard of transformers one way or another. **GPT-3 and BERT** to name a few well known ones :unicorn:. The main idea
is that they showed that you don't have to use recurrent or convolutional layers and that simple architecture coupled with attention is super powerful. It
gave the benefit of **much better long-range dependencies modeling** and the architecture itself is very **parallelizable** (:computer::computer::computer:) which leads to higher compute efficiency!

Here is how their beautifully simple architecture looks like:

<p align="center">
<img src="data/readme_pics/transformer_architecture.PNG" width="350"/>
</p>

## Understanding transformers

This repo is supposed to be a learning resource for understanding transformers as the original transformer by itself is not a SOTA anymore.

For that purpose the code is (hopefully) well commented and I've included the `playground.py` where I've visualized a couple
of concepts which are hard to explain using words but super simple once visualized. So here we go!

### Positional Encodings

Can you parse this one in a glimpse of the eye?

<p align="left">
<img src="data/readme_pics/positional_encoding_formula.PNG"/>
</p>

Neither can I. Running the `visualize_positional_encodings()` function from `playground.py` we get this:

<p align="center">
<img src="data/readme_pics/positional_encoding_visualized.jpg"/>
</p>

Depending on the position of your source/target token you "pick one row of this image" and you add it to it's embedding vector, that's it.
They could also be learned, but it's just more fancy to do it like this, obviously! :nerd_face:

### Custom Learning Rate Schedule

Similarly can you parse this one in `O(1)`?

<p align="left">
<img src="data/readme_pics/lr_formula.PNG"/>
</p>

Noup? So I thought, here it is visualized:

<p align="center">
<img src="data/readme_pics/custom_learning_rate_schedule.PNG"/>
</p>

It's super easy to understand now. Now whether this part was crucial for the success of transformer? I doubt it.
But it's cool and makes things more complicated. :nerd_face:

*Note: model dimension is basically the size of the embedding vector, baseline transformer used 512, the big one 1024*

### Label Smoothing

First time you hear of label smoothing it sounds tough but it's not. You usually set your target vocabulary distribution
to a `one-hot`. Meaning 1 position out of 30k (or whatever your vocab size is) is set to 1. probability and everything else to 0.

<p align="center">
<img src="data/readme_pics/label_smoothing.PNG" width="850"/>
</p>

In label smoothing instead of placing 1. on that particular position you place say 0.9 and you evenly distribute the rest of
the "probability mass" over the other positions 
(that's visualized as a different shade of purple on the image above in a fictional vocab of size 4 - hence 4 columns)

*Note: Pad token's distribution is set to all zeros as we don't want our model to predict those!*

Aside from this repo (well duh) I would highly recommend you go ahead and read [this amazing blog](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar!

## Machine translation

Transformer was originally trained on the NMT (neural machine translation) task on the WMT-14 dataset for English->German
and English->French language pairs. What I did is I trained my models on IWSLT which is much smaller and I used German-English pair,
both directions. I also am planning to train the fully-trained WMT-14 models take a look at [todo](#todo) section.

Anyways! Let's see what this repo can practically do for you? Well it can translate from one human language into another!

Concretely I experimented with **German-English** pair as I speak those 2 so it's easier to debug.

Examples outputs of my German->English model:
* I input `Ich bin ein guter Mensch, denke ich.` and I get out `['<s>', 'I', 'think', 'I', "'m", 'a', 'good', 'person', '.', '</s>']`

Which is actually pretty good!

### Evaluating - BLEU

What is BLEU - short description.

Training the German->English transformer on IWSLT for 20 epochs I got BLEU of .

### Visualizing attention


## HW requirements

This section naturally leads to the next section.

### todos
* Add multi-GPU/multi-node support
* Add beam decoding

### My videos

## Acknoweldgements


## Citation

If you find this code useful for your research, please cite the following:

```
@misc{Gordić2020PyTorchOriginalTransformer,
  author = {Gordić, Aleksa},
  title = {pytorch-original-transformer},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/gordicaleksa/pytorch-original-transformer}},
}
```

## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/gordicaleksa/pytorch-original-transformer/blob/master/LICENCE)