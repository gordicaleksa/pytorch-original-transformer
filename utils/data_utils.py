import time
import os


import torch
from torchtext.data import Dataset, BucketIterator, Field, Example
from torchtext.data.utils import interleave_keys
from torchtext import datasets
import spacy


from .constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


#
# Caching mechanism datasets and functions (you don't need this but it makes things a lot faster!
#


class FastTranslationDataset(Dataset):
    """
        After understanding the source code of torch text's IWSLT, TranslationDataset and Dataset I realized how I
        can make data preparation much faster (tokenization was taking a lot of time and there is no need to redo it
        every time) by using a simple caching mechanism.

        This dataset leverages that caching mechanism which reduced loading time from ~70s -> 2.5s (massive!)

    """

    @staticmethod
    def sort_key(ex):
        # What this does is basically it takes a 16-bit binary representation of lengths and interleaves them.
        # Example: lengths len(ex.src)=5 and len(ex.trg)=3 result in f(101, 011)=100111, 7 and 1 in f(111, 001)=101011
        # It's basically a heuristic that helps the BucketIterator sort bigger batches first
        return interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, cache_path, fields, **kwargs):
        # save_cache interleaves src and trg examples so here we read the cache file having that format in mind
        cached_data = [line.split() for line in open(cache_path, encoding='utf-8')]

        cached_data_src = cached_data[0::2]  # Even lines contain source examples
        cached_data_trg = cached_data[1::2]  # Odd lines contain target examples

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
    """
        Just a wrapper around the FastTranslationDataset.

    """

    @classmethod
    def get_train_and_val_datasets(cls, train_cache_path, val_cache_path, fields, **kwargs):

        train_dataset = cls(train_cache_path, fields, **kwargs)
        val_dataset = cls(val_cache_path, fields, **kwargs)

        return train_dataset, val_dataset


def save_cache(cache_path, dataset):
    with open(cache_path, 'w', encoding='utf-8') as cache_file:
        # Interleave source and target tokenized examples, source is on even lines, target is on odd lines
        for ex in dataset.examples:
            cache_file.write(' '.join(ex.src) + '\n')
            cache_file.write(' '.join(ex.trg) + '\n')


#
# End of caching mechanism utilities
#


# todo: add BPE
# todo: try first with this smaller dataset latter add support for WMT-14 as well
def build_datasets_and_vocabs(dataset_path, use_caching_mechanism=True):
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    # batch first set to ture as my transformer is expecting that format (that's consistent with the format
    # used in  computer vision), namely (B, C, H, W) -> batch size, number of channels, height and width
    src_field_processor = Field(tokenize=tokenize_de, pad_token=PAD_TOKEN, batch_first=True)
    trg_field_processor = Field(tokenize=tokenize_en, init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, batch_first=True)

    fields = [('src', src_field_processor), ('trg', trg_field_processor)]
    MAX_LEN = 100  # filter out examples that have more than MAX_LEN tokens
    filter_pred = lambda x: len(x.src) <= MAX_LEN and len(x.trg) <= MAX_LEN

    # Only call once the splits function it is super slow as it constantly has to redo the tokenization
    train_cache_path = os.path.join(dataset_path, 'train_cache.csv')
    val_cache_path = os.path.join(dataset_path, 'val_cache.csv')
    test_cache_path = os.path.join(dataset_path, 'test_cache.csv')

    # This simple caching mechanism gave me ~30x speedup on my machine! From ~70s -> ~2.5s!
    ts = time.time()
    if not use_caching_mechanism or not (os.path.exists(train_cache_path) and os.path.exists(val_cache_path)):
        # dataset objects have a list of examples where example is simply an empty Python Object that has
        # .src and .trg attributes which contain a tokenized list of strings (created by tokenize_en and tokenize_de).
        # It's that simple, we can consider our datasets as a table with 2 columns 'src' and 'trg'
        # each containing fields with tokenized strings from source and target languages
        train_dataset, val_dataset, test_dataset = datasets.IWSLT.splits(
            exts=('.de', '.en'),
            fields=fields,
            root=dataset_path,
            filter_pred=filter_pred
        )

        save_cache(train_cache_path, train_dataset)
        save_cache(val_cache_path, val_dataset)
        save_cache(test_cache_path, test_dataset)
    else:
        # it's actually better to load from cache as we'll get rid of '\xa0', '\xa0 ' and '\x85' unicode characters
        # which we don't need and which SpaCy unfortunately includes as tokens.
        train_dataset, val_dataset = DatasetWrapper.get_train_and_val_datasets(
            train_cache_path,
            val_cache_path,
            fields,
            filter_pred=filter_pred
        )

    print(f'Time it took to prepare the data: {time.time() - ts:3f} seconds.')

    MIN_FREQ = 2
    # __getattr__ implementation in the base Dataset class enables us to call .src on Dataset objects even though
    # we only have a list of examples in the Dataset object and the example itself had .src attribute.
    # Implementation will yield examples and call .src/.trg attributes on them (and those contain tokenized lists)
    src_field_processor.build_vocab(train_dataset.src, min_freq=MIN_FREQ)
    trg_field_processor.build_vocab(train_dataset.trg, min_freq=MIN_FREQ)

    return train_dataset, val_dataset, src_field_processor, trg_field_processor


global longest_src_sentence, longest_trg_sentence


def batch_size_fn(new_example, count, sofar):
    """
        If we use this function in the BucketIterator the batch_size is no longer the number of examples/sentences
        in a batch but a number of tokens in a batch - which allows us to max out VRAM on a given GPU.

        Example: if we don't use this function and we set batch size to say 10 we will sometimes end up with
        a tensor of size (10, 100) because the longest sentence had a size of 100 tokens but other times we'll end
        up with a size of (10, 5) because the longest sentence had only 5 tokens!

        With this function what we do is we specify that source and target tensors can't go over a certain number
        of tokens like 1000. So usually either source or target tensors will contain around 1000 tokens and
        in worst case both will be really close to a 1000 tokens each. If that is still below max VRAM availabe on
        the system we're using the max potential of our GPU w.r.t. VRAM.

        Note: to understand this function you unfortunately would probably have to dig deeper into torch text's
        source code.

    """
    global longest_src_sentence, longest_trg_sentence

    if count == 1:
        longest_src_sentence = 0
        longest_trg_sentence = 0

    longest_src_sentence = max(longest_src_sentence, len(new_example.src))
    # 2 because of start/end of sentence tokens (<s> and </s>)
    longest_trg_sentence = max(longest_trg_sentence, len(new_example.trg) + 2)

    num_of_tokens_in_src_tensor = count * longest_src_sentence
    num_of_tokens_in_trg_tensor = count * longest_trg_sentence

    return max(num_of_tokens_in_src_tensor, num_of_tokens_in_trg_tensor)


# https://github.com/pytorch/text/issues/536#issuecomment-719945594 <- there is a "bug" in BucketIterator i.e. it's
# description is misleading as it won't group examples of similar length unless you set sort_within_batch to True!
def get_data_loaders(dataset_path, batch_size, device):
    train_dataset, val_dataset, src_field_processor, trg_field_processor = build_datasets_and_vocabs(dataset_path)

    # using default sorting function which
    train_token_ids_loader, val_token_ids_loader = BucketIterator.splits(
     datasets=(train_dataset, val_dataset),
     batch_size=batch_size,
     device=device,
     sort_within_batch=True,
     batch_size_fn=batch_size_fn
    )

    return train_token_ids_loader, val_token_ids_loader, src_field_processor, trg_field_processor


def build_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id):
    batch_size = src_token_ids_batch.shape[0]

    # src_mask shape = (B, 1, 1, S) check out attention function in transformer_model.py where masks are applied
    # src_mask only masks pad tokens as we want to ignore their representations (no information in there...)
    src_mask = (src_token_ids_batch != pad_token_id).view(batch_size, 1, 1, -1)
    num_src_tokens = torch.sum(src_mask.long())

    return src_mask, num_src_tokens


def build_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id):
    batch_size = trg_token_ids_batch.shape[0]
    device = trg_token_ids_batch.device

    # Same as src_mask but we additionally want to mask tokens from looking forward into the future tokens
    # Note: wherever the mask value is true we want to attend to that token, otherwise we mask (ignore) it.
    sequence_length = trg_token_ids_batch.shape[1]  # trg_token_ids shape = (B, T) where T max trg token-sequence length
    trg_padding_mask = (trg_token_ids_batch != pad_token_id).view(batch_size, 1, 1, -1)  # shape = (B, 1, 1, T)
    trg_no_look_forward_mask = torch.triu(torch.ones((1, 1, sequence_length, sequence_length), device=device) == 1).transpose(2, 3)

    # logic AND operation (both padding mask and no-look-forward must be true to attend to a certain target token)
    trg_mask = trg_padding_mask & trg_no_look_forward_mask  # final shape = (B, 1, T, T)
    num_trg_tokens = torch.sum(trg_padding_mask.long())

    return trg_mask, num_trg_tokens


def build_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch, pad_token_id, device):
    src_mask, num_src_tokens = build_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id)
    trg_mask, num_trg_tokens = build_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id)

    return src_mask, trg_mask, num_src_tokens, num_trg_tokens


def fetch_src_and_trg_batches(token_ids_batch):
    src_token_ids_batch, trg_token_ids_batch = token_ids_batch.src, token_ids_batch.trg

    # Target input should be shifted by 1 compared to the target output tokens
    # Example: if we had a sentence like: [<s>,what,is,up,</s>] then to train the NMT model what we do is we pass
    # [<s>,what,is,up] to the input as set [what,is,up,</s>] as the expected output.
    trg_token_ids_batch_input = trg_token_ids_batch[:, :-1]

    # We reshape from (B, S) into (BxS, 1) as that's the the shape expected by LabelSmoothing which will produce
    # the shape (BxS, V) where V is the target vocab size which is the same shape as the one that comes out
    # from the transformer so we can directly pass them into the KL divergence loss
    trg_token_ids_batch_gt = trg_token_ids_batch[:, 1:].reshape(-1, 1)

    return src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt


#
# Everything below is for testing purposes only - feel free to ignore
#


def sample_text_from_loader(src_field_processor, trg_field_processor, token_ids_loader, num_samples=2, sample_src=True, sample_trg=True, show_padded=False):
    assert sample_src or sample_trg, f'Either src or trg or both must be enabled.'

    for b_idx, token_ids_batch in enumerate(token_ids_loader):
        if b_idx == num_samples:  # Number of sentence samples to print
            break

        print('*' * 5)
        if sample_src:
            print("Source text:", end="\t")
            for token_id in token_ids_batch.src[0]:  # print only the first example from the batch
                src_token = src_field_processor.vocab.itos[token_id]

                if src_token == PAD_TOKEN and not show_padded:
                    continue

                print(src_token, end=" ")
            print()

        if sample_trg:
            print("Target text:", end="\t")
            for token_id in token_ids_batch.trg[0]:
                trg_token = trg_field_processor.vocab.itos[token_id]

                if trg_token == PAD_TOKEN and not show_padded:
                    continue

                print(trg_token, end=" ")
            print()


if __name__ == "__main__":
    # To run this delete the dot from from .constants import - not the most elegant solution but it works
    # without me having to add sys.path stuff, if you have a more elegant solution please open an issue <3
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = os.path.join(os.path.dirname(__file__), os.pardir, '.data')
    train_token_ids_loader, val_token_ids_loader, src_field_processor, trg_field_processor = get_data_loaders(dataset_path, batch_size, device)

    # Verify that the mask logic is correct
    pad_token_id = src_field_processor.vocab.stoi[PAD_TOKEN]
    for batch in train_token_ids_loader:
        # Visually inspect that masks make sense
        src_padding_mask, trg_mask, num_src_tokens, num_trg_tokens = build_masks_and_count_tokens(batch.src, batch.trg, pad_token_id, device)
        break

    # Check vocab size
    print(f'Source vocabulary size={len(src_field_processor.vocab)}')
    print(f'Target vocabulary size={len(trg_field_processor.vocab)}')

    # Show text from token loader
    sample_text_from_loader(src_field_processor, trg_field_processor, train_token_ids_loader)

