# todo: step1 - figure out how the first part works (toy example) (DONE)
# todo: step2 - rewrite the whole thing it's not a good design
# todo: step3 - add real example support (rewrite)
# todo: step4 - add beam search and other missing components
# todo: step5 - train the model and report BLEU
# todo: step6 - write README and open-source

# todo: take a look at the naming used in the original paper
# todo: use built-in blocks but implement them myself also
# todo: a must: implement simple, dedicated multi-headed SELF-attention
# todo: add a jupyter notebook
# todo: create this in a similar fashion to GANs repo, things I've modified, etc.

from torch import nn

criterion = nn.KLDivLoss(reduction='batchmean')