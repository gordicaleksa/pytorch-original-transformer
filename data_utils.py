import time


import torch
from torchtext.data import Iterator, BucketIterator, Field
from torchtext import datasets
import spacy


from constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


# todo: once I process the data once use cache files (https://github.com/bentrevett/pytorch-sentiment-analysis/issues/6)
# otherwise it's super slow (~60 seconds on my machine)
# todo: see whether I should use tgt or trg, also pad_idx or pad_token_idx
def build_datasets_and_vocabs():
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    SRC = Field(tokenize=tokenize_de, pad_token=PAD_TOKEN, batch_first=True)
    TGT = Field(tokenize=tokenize_en, init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, batch_first=True)

    MAX_LEN = 100
    ts = time.time()
    # todo: try first with this smaller dataset latter add support for WMT-14 as well
    train_dataset, val_dataset, test_dataset = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(SRC, TGT),
                                             filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(
                                                 vars(x)['trg']) <= MAX_LEN)
    print(f'Time it took to load the data: {time.time() - ts:3f} seconds.')

    MIN_FREQ = 2
    ts = time.time()
    SRC.build_vocab(train_dataset.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train_dataset.trg, min_freq=MIN_FREQ)
    print(f'Time it took to build vocabs: {time.time() - ts:3f} seconds.')
    return train_dataset, val_dataset, test_dataset, SRC, TGT


def get_data_loaders(batch_size=8):
    train_dataset, val_dataset, test_dataset, SRC, TGT = build_datasets_and_vocabs()

    # todo: figure out how to set the optimal batch size
    # todo: verify that BucketIterator is really minimizing the number of pad tokens
    train_token_ids_loader, val_token_ids_loader = BucketIterator.splits(
     datasets=(train_dataset, val_dataset),
     batch_sizes=(batch_size, batch_size),
     device=0,
     # sort_key=lambda x: len(vars(x)['src']), # the BucketIterator needs to be told what function it should use to group the data.
    )
    test_token_ids_loader = Iterator(test_dataset, batch_size=64, device=0, sort=False, sort_within_batch=False, repeat=False)

    return train_token_ids_loader, val_token_ids_loader, test_token_ids_loader, SRC, TGT


def sample_text_from_loader(SRC, TGT, token_ids_loader, num_samples=2, sample_src=True, sample_tgt=True, show_padded=False):
    assert sample_src or sample_tgt, f'Either src or tgt or both must be enabled.'

    for b_idx, batch in enumerate(token_ids_loader):
        if b_idx == num_samples:
            break

        print('*' * 5)
        if sample_src:
            print("Source text:", end="\t")
            for token_id in batch.src[0]:
                src_token = SRC.vocab.itos[token_id]
                if src_token == PAD_TOKEN and not show_padded:
                    continue
                print(src_token, end=" ")
            print()

        if sample_tgt:
            print("Target text:", end="\t")
            for token_id in batch.trg[0]:
                tgt_token = TGT.vocab.itos[token_id]
                if tgt_token == PAD_TOKEN and not show_padded:
                    continue
                print(tgt_token, end=" ")
            print()


def build_masks_and_count_tokens(src_token_ids_batch, tgt_token_ids_batch, padding_token_id):
    batch_size = src_token_ids_batch.shape[0]

    # src_mask shape = (B, 1, 1, S) check out attention function in transformer_model.py where masks are applied
    # src_mask only masks pad tokens as we want to ignore their representations (no information there...)
    src_mask = (src_token_ids_batch != padding_token_id).view(batch_size, 1, 1, -1)
    num_src_tokens = torch.sum(src_mask.long())

    # Same as src_mask but we additionally want to mask tokens from looking forward into the future tokens
    # Note: wherever the mask value is true we want to attend to that token, otherwise we mask (ignore) it.
    sequence_length = tgt_token_ids_batch.shape[1]
    tgt_padding_mask = (tgt_token_ids_batch != padding_token_id).view(batch_size, 1, 1, -1)
    tgt_no_look_forward_mask = torch.triu(torch.ones((1, 1, sequence_length, sequence_length)) == 1).transpose(2, 3)

    # logic AND operation (both padding mask and no-look-forward must be true to attend to a certain token)
    tgt_mask = tgt_padding_mask & tgt_no_look_forward_mask
    num_tgt_tokens = torch.sum(tgt_padding_mask.long())

    return src_mask, tgt_mask, num_src_tokens, num_tgt_tokens


def fetch_src_and_tgt_batches(token_ids_batch):
    src_token_ids_batch, tgt_token_ids_batch = token_ids_batch.src, token_ids_batch.trg

    # Input should be shifted by 1 compared to the output tokens
    # Example: if we had a sentence like: <s>,what,is,up,</s> than to train the NMT model what we do is we pass
    # <s>,what,is,up to the input as set what,is,up,</s> as the expected output.
    tgt_token_ids_batch_input = tgt_token_ids_batch[:, :-1]

    # We reshape from (B, S) into (BxS, 1) as that's the the shape expected by LabelSmoothing which will produce
    # the shape (BxS, V) where V is the target vocab size which is the same shape that comes out from the transformer
    # so we can directly pass them into KL divergence loss
    tgt_token_ids_batch_gt = tgt_token_ids_batch[:, 1:].reshape(-1, 1)

    return src_token_ids_batch, tgt_token_ids_batch_input, tgt_token_ids_batch_gt


# For testing purposes feel free to ignore
if __name__ == "__main__":
    train_token_ids_loader, val_token_ids_loader, test_token_ids_loader, SRC, TGT = get_data_loaders()

    for batch in train_token_ids_loader:
        src_padding_mask, tgt_mask, num_src_tokens, num_tgt_tokens = build_masks_and_count_tokens(batch.src, batch.trg, 1)

    print(f'Source vocabulary size={len(SRC.vocab)}')
    print(f'Target vocabulary size={len(TGT.vocab)}')

    sample_text_from_loader(SRC, TGT, train_token_ids_loader)

