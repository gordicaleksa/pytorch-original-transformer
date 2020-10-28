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


from constants import *
from data_utils import get_data_loaders, build_masks_and_count_tokens
from transformer_model import Transformer


if __name__ == "__main__":
    train_token_ids_loader, val_token_ids_loader, test_token_ids_loader, SRC, TGT = get_data_loaders()

    baseline_transformer = Transformer(
        model_dimension=BASELINE_MODEL_DIMENSION,
        src_vocab_size=len(SRC.vocab),
        tgt_vocab_size=len(TGT.vocab),
        number_of_heads=BASELINE_MODEL_NUMBER_OF_HEADS,
        number_of_layers=BASELINE_MODEL_NUMBER_OF_LAYERS,
        dropout_probability=BASELINE_MODEL_DROPOUT_PROB
    )

    loss_fn = nn.KLDivLoss(reduction='batchmean')

    for token_ids_batch in train_token_ids_loader:
        src_token_ids_batch, tgt_token_ids_batch = token_ids_batch.src, token_ids_batch.trg
        src_padding_mask, tgt_mask, num_src_tokens, num_tgt_tokens = build_masks_and_count_tokens(src_token_ids_batch, tgt_token_ids_batch)
