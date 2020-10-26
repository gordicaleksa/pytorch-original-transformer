from torchtext.data import Iterator, BucketIterator, Field
from torchtext import datasets


import spacy
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
SRC = Field(tokenize=tokenize_de, pad_token=BLANK_WORD, batch_first=True)
TGT = Field(tokenize=tokenize_en, init_token=BOS_WORD,
                 eos_token=EOS_WORD, pad_token=BLANK_WORD, batch_first=True)

MAX_LEN = 100
# todo: try first with this smaller dataset latter add support for WMT-14 as well
train, val, test = datasets.IWSLT.splits(
    exts=('.de', '.en'), fields=(SRC, TGT),
    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and
                          len(vars(x)['trg']) <= MAX_LEN)
MIN_FREQ = 2
SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

print(type(SRC), type(train))


train_iter, val_iter = BucketIterator.splits(
 (train, val), # we pass in the datasets we want the iterator to draw data from
 batch_sizes=(64, 64),
 device=0,
 sort_key=lambda x: len(x.comment_text), # the BucketIterator needs to be told what function it should use to group the data.
)
test_iter = Iterator(test, batch_size=64, device=-1, sort=False, sort_within_batch=False, repeat=False)

print(SRC.vocab.stoi[BLANK_WORD])
print(TGT.vocab.stoi[BOS_WORD], TGT.vocab.stoi[EOS_WORD], TGT.vocab.stoi[BLANK_WORD])

for i in range(100):
    print(SRC.vocab.itos[i])
    print(TGT.vocab.itos[i])


for batch in train_iter:
    print(type(batch))
    print(batch.src[0])
    print(batch.trg[0])

    print("Translation:", end="\t")
    for i in batch.src[0]:
        sym = SRC.vocab.itos[i]
        print(sym, end=" ")

    print("Target:", end="\t")
    for i in batch.trg[0]:
        sym = TGT.vocab.itos[i]
        print(sym, end=" ")
    print()
