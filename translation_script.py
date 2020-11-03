import argparse


import torch
from torchtext.data import Example


from models.definitions.transformer_model import Transformer
from utils.data_utils import build_datasets_and_vocabs, build_masks_and_count_tokens_src, build_masks_and_count_tokens_trg
from utils.constants import *
from utils.visualization_utils import visualize_attention


# todo: add beam decoding mechanism
def beam_decoding():
    print('todo')


def greedy_decoding(baseline_transformer, src_representations_batch, src_mask, trg_field_processor, max_target_tokens=60):
    device = next(baseline_transformer.parameters()).device
    pad_token_id = trg_field_processor.vocab.stoi[PAD_TOKEN]

    target_sentence_tokens = [BOS_TOKEN]  # initial prompt - beginning/start of the sentence token
    trg_token_ids_batch = torch.tensor([[trg_field_processor.vocab.stoi[token] for token in target_sentence_tokens]], device=device)

    while True:
        trg_mask, _ = build_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id)
        predicted_log_distributions = baseline_transformer.decode(trg_token_ids_batch, src_representations_batch, trg_mask, src_mask)

        # This is the "greedy" part:
        # We find the index of the target token with highest probability and discard every other possibility
        most_probable_word_index = torch.argmax(predicted_log_distributions[-1]).cpu().numpy()
        # Find the target token associated with this index
        predicted_word = trg_field_processor.vocab.itos[most_probable_word_index]

        target_sentence_tokens.append(predicted_word)
        if predicted_word == EOS_TOKEN or len(target_sentence_tokens) == max_target_tokens:
            break

        # Prepare the input for the next iteration
        trg_token_ids_batch = torch.tensor([[trg_field_processor.vocab.stoi[token] for token in target_sentence_tokens]], device=device)

    return target_sentence_tokens


# Super easy to add translation for a batch of sentences passed as a .txt file for example
def translate_a_single_sentence(translation_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

    # Step 1: Prepare the field processor (tokenizer, numericalizer)
    _, _, src_field_processor, trg_field_processor = build_datasets_and_vocabs(translation_config['dataset_path'])
    assert src_field_processor.vocab.stoi[PAD_TOKEN] == trg_field_processor.vocab.stoi[PAD_TOKEN]
    pad_token_id = src_field_processor.vocab.stoi[PAD_TOKEN]  # needed for constructing masks

    # Step 2: Prepare the model
    baseline_transformer = Transformer(
        model_dimension=BASELINE_MODEL_DIMENSION,
        src_vocab_size=len(src_field_processor.vocab),
        trg_vocab_size=len(trg_field_processor.vocab),
        number_of_heads=BASELINE_MODEL_NUMBER_OF_HEADS,
        number_of_layers=BASELINE_MODEL_NUMBER_OF_LAYERS,
        dropout_probability=BASELINE_MODEL_DROPOUT_PROB
    ).to(device)

    model_path = os.path.join(BINARIES_PATH, translation_config['model_name'])
    assert os.path.exists(model_path), f'Could not find the model {model_path}. You first need to train the transformer.'

    model_state = torch.load(model_path)
    baseline_transformer.load_state_dict(model_state["state_dict"], strict=True)
    baseline_transformer.eval()

    # Step 3: Prepare the input sentence
    source_sentence = translation_config['german_sentence']
    ex = Example.fromlist([source_sentence], fields=[('src', src_field_processor)])  # tokenize the sentence

    source_sentence_tokens = ex.src
    print(f'Source sentence tokens = {source_sentence_tokens}')

    # Numericalize and convert to cuda tensor
    src_token_ids_batch = src_field_processor.process([source_sentence_tokens], device)

    # Decoding could be further optimized to cache old token activations because they can't look ahead and so
    # adding a newly predicted token won't change old token's activations.
    #
    # Example: we input <s> and do a forward pass. We get intermediate activations for <s> and at the output at position
    # 0, after the doing linear layer we get e.g. token <I>. Now we input <s>,<I> but <s>'s activations will remain
    # the same. Similarly say we now got <am> at output position 1, in the next step we input <s>,<I>,<am> and so <I>'s
    # activations will remain the same as it only looks at/attends to itself and to <s> and so forth.

    with torch.no_grad():
        # Step 4: Optimization - compute the source token representations only once
        src_mask, _ = build_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id)
        src_representations_batch = baseline_transformer.encode(src_token_ids_batch, src_mask)

        # Step 5: Decoding process
        target_sentence_tokens = greedy_decoding(baseline_transformer, src_representations_batch, src_mask, trg_field_processor)
        print(f'Translation | Target sentence tokens = {target_sentence_tokens}')

        # Step 6: Potentially visualize the encoder/decoder attention weights
        if translation_config['visualize_attention']:
            visualize_attention(baseline_transformer, source_sentence_tokens, target_sentence_tokens)


if __name__ == "__main__":
    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--german_sentence", type=str, help="German sentence to translate into English", default="Ich bin ein guter Mensch, denke ich.")
    parser.add_argument("--model_name", type=str, help="transformer model name", default=r'transformer_000000.pth')
    parser.add_argument("--dataset_path", type=str, help='save dataset to this path', default=os.path.join(os.path.dirname(__file__), '.data'))

    parser.add_argument("--visualize_attention", type=bool, help="should visualize encoder/decoder attention", default=True)
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    translation_config = dict()
    for arg in vars(args):
        translation_config[arg] = getattr(args, arg)

    # Translate the given german sentence
    translate_a_single_sentence(translation_config)
