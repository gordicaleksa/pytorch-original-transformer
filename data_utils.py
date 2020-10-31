import time
import os


import torch
from torchtext.data import Dataset, BucketIterator, Field, Example
from torchtext.data.utils import interleave_keys
from torchtext import datasets
import spacy


from constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


class FastTranslationDataset(Dataset):

    @staticmethod
    def sort_key(ex):
        return interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, cache_path, fields, **kwargs):
        cached_data = [line.split() for line in open(cache_path, encoding='utf-8')]
        cached_data_src = cached_data[0::2]
        cached_data_trg = cached_data[1::2]
        assert len(cached_data_src) == len(cached_data_trg), f'Source and target data should be of the same length.'

        examples = []
        for src_tokenized_data, trg_tokenized_data in zip(cached_data_src, cached_data_trg):
            ex = Example()

            setattr(ex, 'src', src_tokenized_data)
            setattr(ex, 'trg', trg_tokenized_data)

            examples.append(ex)

        # Call the parent class Dataset's constructor
        super().__init__(examples, fields, **kwargs)


class DatasetWrapper(FastTranslationDataset):

    @classmethod
    def get_train_and_val_datasets(cls, train_cache_path, val_cache_path, fields, **kwargs):

        train_dataset = cls(train_cache_path, fields, **kwargs)
        val_dataset = cls(val_cache_path, fields, **kwargs)

        return train_dataset, val_dataset


def save_cache(cache_path, dataset):
    with open(cache_path, 'w', encoding='utf-8') as train_cache:
        for ex in dataset.examples:
            train_cache.write(' '.join(ex.src) + '\n')
            train_cache.write(' '.join(ex.trg) + '\n')


# todo: add BPE
# todo: see whether I should use tgt or trg, also pad_idx or pad_token_idx
def build_datasets_and_vocabs():
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    SRC = Field(tokenize=tokenize_de, pad_token=PAD_TOKEN, batch_first=True)
    TRG = Field(tokenize=tokenize_en, init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, batch_first=True)
    fields = [('src', SRC), ('trg', TRG)]
    filter_pred = lambda x: len(x.src) <= MAX_LEN and len(x.trg) <= MAX_LEN

    MAX_LEN = 100
    ts = time.time()

    # Only call once the splits function it is super slow as it constantly has to redo the tokenization
    root = '.data'
    train_cache_path = os.path.join(root, 'train_cache.csv')
    val_cache_path = os.path.join(root, 'val_cache.csv')
    test_cache_path = os.path.join(root, 'test_cache.csv')

    # This simple caching mechanism gave me ~30x speedup on my machine! From ~70s -> ~2.5s!
    if not (os.path.exists(train_cache_path) and os.path.exists(val_cache_path)):
        # it's got a list of examples where example is simply an object that has .src and .trg attributes which
        # contain a tokenized list of strings using the provided tokenizer functions (tokenize_en, tokenize_de).
        # It's that simple. It also internally save self.fields = {'src': SRC, 'trg': TRG} that's it. i.e. we can
        # consider our datasets as 2 columns src and trg each containing fields with tokenized strings
        # todo: try first with this smaller dataset latter add support for WMT-14 as well
        train_dataset, val_dataset, test_dataset = datasets.IWSLT.splits(
            exts=('.de', '.en'),
            fields=fields,
            filter_pred=filter_pred
        )

        save_cache(train_cache_path, train_dataset)
        save_cache(val_cache_path, val_dataset)
        save_cache(test_cache_path, test_dataset)
    else:
        # it's actually a bit better to load from cache as we'll get rid of '\xa0', '\xa0 ' and '\x85' unicode characters
        # which we don't need
        train_dataset, val_dataset = DatasetWrapper.get_train_and_val_datasets(train_cache_path, val_cache_path, fields, filter_pred=filter_pred)

    print(f'Time it took to load the data: {time.time() - ts:3f} seconds.')

    MIN_FREQ = 2
    ts = time.time()
    # __getattr__ enables us to call .src even though we only have a list of examples, it will yield examples and call
    # .src/.trg attributes on them which contain tokenized lists as explained earlier
    SRC.build_vocab(train_dataset.src, min_freq=MIN_FREQ)
    TRG.build_vocab(train_dataset.trg, min_freq=MIN_FREQ)
    print(f'Time it took to build vocabs: {time.time() - ts:3f} seconds.')
    return train_dataset, val_dataset, SRC, TRG


def get_data_loaders(batch_size, device):
    train_dataset, val_dataset, SRC, TRG = build_datasets_and_vocabs()
    # todo: figure out how to set the optimal batch size
    # todo: verify that BucketIterator is really minimizing the number of pad tokens
    # using default sorting function which
    train_token_ids_loader, val_token_ids_loader = BucketIterator.splits(
     datasets=(train_dataset, val_dataset),
     batch_size=batch_size,
     device=device,
    )
    # todo: seems like sort_within_batch is a must and using the non-default batch_size_fn (just counts the num of
    #  examples when chunking) is a smart idea
    # todo: bug in pytorch if sort_within_batch is not set to True it's not grouping according to length!?
    # todo: step into code from my torch text version
    return train_token_ids_loader, val_token_ids_loader, SRC, TRG


def sample_text_from_loader(SRC, TRG, token_ids_loader, num_samples=2, sample_src=True, sample_trg=True, show_padded=False):
    assert sample_src or sample_trg, f'Either src or trg or both must be enabled.'

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

        if sample_trg:
            print("Target text:", end="\t")
            for token_id in batch.trg[0]:
                trg_token = TRG.vocab.itos[token_id]
                if trg_token == PAD_TOKEN and not show_padded:
                    continue
                print(trg_token, end=" ")
            print()


def build_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch, padding_token_id, device):
    batch_size = src_token_ids_batch.shape[0]

    # src_mask shape = (B, 1, 1, S) check out attention function in transformer_model.py where masks are applied
    # src_mask only masks pad tokens as we want to ignore their representations (no information there...)
    src_mask = (src_token_ids_batch != padding_token_id).view(batch_size, 1, 1, -1)
    num_src_tokens = torch.sum(src_mask.long())

    # Same as src_mask but we additionally want to mask tokens from looking forward into the future tokens
    # Note: wherever the mask value is true we want to attend to that token, otherwise we mask (ignore) it.
    sequence_length = trg_token_ids_batch.shape[1]
    trg_padding_mask = (trg_token_ids_batch != padding_token_id).view(batch_size, 1, 1, -1)
    trg_no_look_forward_mask = torch.triu(torch.ones((1, 1, sequence_length, sequence_length), device=device) == 1).transpose(2, 3)

    # logic AND operation (both padding mask and no-look-forward must be true to attend to a certain token)
    trg_mask = trg_padding_mask & trg_no_look_forward_mask
    num_trg_tokens = torch.sum(trg_padding_mask.long())

    return src_mask, trg_mask, num_src_tokens, num_trg_tokens


def fetch_src_and_trg_batches(token_ids_batch):
    src_token_ids_batch, trg_token_ids_batch = token_ids_batch.src, token_ids_batch.trg

    # Input should be shifted by 1 compared to the output tokens
    # Example: if we had a sentence like: <s>,what,is,up,</s> than to train the NMT model what we do is we pass
    # <s>,what,is,up to the input as set what,is,up,</s> as the expected output.
    trg_token_ids_batch_input = trg_token_ids_batch[:, :-1]

    # We reshape from (B, S) into (BxS, 1) as that's the the shape expected by LabelSmoothing which will produce
    # the shape (BxS, V) where V is the target vocab size which is the same shape that comes out from the transformer
    # so we can directly pass them into KL divergence loss
    trg_token_ids_batch_gt = trg_token_ids_batch[:, 1:].reshape(-1, 1)

    return src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt


# For testing purposes feel free to ignore
if __name__ == "__main__":
    # train_token_ids_loader, val_token_ids_loader, test_token_ids_loader, SRC, TRG = get_data_loaders()
    #
    # for batch in train_token_ids_loader:
    #     src_padding_mask, trg_mask, num_src_tokens, num_trg_tokens = build_masks_and_count_tokens(batch.src, batch.trg, 1)
    #
    # print(f'Source vocabulary size={len(SRC.vocab)}')
    # print(f'Target vocabulary size={len(TRG.vocab)}')
    #
    # sample_text_from_loader(SRC, TRG, train_token_ids_loader)

    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')


    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]


    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]


    SRC = Field(tokenize=tokenize_de, pad_token=PAD_TOKEN, batch_first=True)
    TRG = Field(tokenize=tokenize_en, init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, batch_first=True)

    src_line = 'Ich bin.'
    trg_line = 'I am'
    fields = [('src', SRC), ('trg', TRG)]
    ex = Example.fromlist([src_line, trg_line], fields)
    print(ex.src, ex.trg)
