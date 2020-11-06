## The Original Transformer (PyTorch) :computer: = :rainbow:
This repo contains PyTorch implementation of the original transformer paper (:link: [Vaswani et al.](https://arxiv.org/pdf/1706.03762.pdf)). <br/>
It's aimed at making it **easy for beginners** to start playing and learning about transformers. <br/>

**Important note:** I'll be adding a jupyter notebook soon as well!

## Table of Contents
  * [What are transformers?](#what-are-transformers)
  * [Understanding transformers](#understanding-transformers)
  * [Machine translation](#machine-translation)
  * [Setup](#setup)
  * [Usage](#usage)
  * [HW requirements for training](#hardware-requirements)

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

Transformer was originally trained on the NMT (neural machine translation) task on the WMT-14 dataset for:
* English to German translations (achieved 28.4 [BLEU score](https://en.wikipedia.org/wiki/BLEU))
* English to French translations (achieved 41.8 BLEU score)
 
What I did (for now) is I trained my models on IWSLT which is much smaller and I trained my models to translate
from English to German and vice versa, as I speak those 2 so it's easier to debug.

I will train WMT-14 models soon, take a look at the [todos](#todos) section.

Anyways! Let's see what this repo can practically do for you! Well it can translate from one human language into another!

Some short outputs from my German to English model:
I input: `Ich bin ein guter Mensch, denke ich.` <br/>
and I get out: `['<s>', 'I', 'think', 'I', "'m", 'a', 'good', 'person', '.', '</s>']`

Which is actually pretty good!

There are of course failure cases like:
I input: `Ich bin ein Berliner` <br/>
and I get back: ``

Similarly for the English2German model, I'll link fully trained models down in the [usage](#usage) section.

## Setup



## Usage

link to my models

### Evaluating NMT models (BLEU metric)

BLEU is an n-gram based metric for quantitatively evaluating the quality of machine translation models.

Training the German->English transformer on IWSLT for 20 epochs I got BLEU of .

Initialization matters! Show BLEU curves from my Azure ML runs: 

### Visualizing attention

You can use the `translation_script.py` and set the `--visualize_attention` to True to additionally understand what your
model was "paying attention to" in the source and target sentences.

Here are the attentions I get for the input sentence `Ich bin ein guter Mensch, denke ich.`.

<p align="center">
<img src="data/readme_pics/attention_enc_self.PNG" width="850"/>
</p>

These belong to layer 4 of the encoder. You can see all of the 8 multi-head attention heads.

<p align="center">
<img src="data/readme_pics/attention_dec_self.PNG" width="850"/>
</p>

And this one belongs to layer 1 of the decoder. You can notice an interesting **triangular pattern** which 
comes from the fact that target tokens can't look ahead!

The 3rd type of MHA (multi-head attention) module is the source attending one but it looks similar to the
plot you saw in the encoder.

## Hardware requirements

You really need a decent hardware if you wish to train the transformer on the **WMT-14** dataset.

The authors took:
* **12h on 8 P100 GPUs** to train the baseline model and **3.5 days** to train the big one.

If my calculations are right that amounts to ~19 epochs (100k steps, each step had ~25000 tokens and WMT-14 has ~130M src/trg tokens)
for the baseline and 3x that for the big one (300k steps).

On the other hand it's much more feasible to train the model on the **IWSLT** dataset. It took me:
* 13.2 min/epoch (1500 token batch) on my RTX 2080 machine (8 GBs of VRAM)
* ~34 min/epoch (1500 token batch) on Azure ML's K80s (24 GBs of VRAM)

I could have pushed K80s to 3500+ tokens/batch but had some CUDA out of memory problems.

### Todos:

Finally there are a couple more todos which I'll hopefully add really soon:
* Multi-GPU/multi-node training support (so that you can train a model on WMT-14 for 19 epochs)
* Beam decoding (turns out it's not that easy to implement this one!)
* BPE and shared source-target vocab

## Video learning material

I also made a video covering how I approached learning transformers, you can check it out on [my YouTube channel:](https://www.youtube.com/watch?v=bvBK-coXf9I&ab_channel=TheAIEpiphany)

<p align="left">
<a href="https://www.youtube.com/watch?v=bvBK-coXf9I" target="_blank"><img src="https://img.youtube.com/vi/bvBK-coXf9I/0.jpg" 
alt="NST Intro" width="480" height="360" border="10" /></a>
</p>

## Acknowledgements

I found these resources useful (while developing this one):

* [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
* [PyTorch official implementation](https://github.com/pytorch/pytorch/blob/187e23397c075ec2f6e89ea75d24371e3fbf9efa/torch/nn/modules/transformer.py)

I found lots of inspiration for the model design in the The Annotated Transformer but I found it hard to understand, and
it had some bugs. It was mainly written with researchers in mind. Hopefully this repo opens up
the understanding of transformers to the common folk as well! :nerd_face:

## Citation

If you find this code useful, please cite the following:

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