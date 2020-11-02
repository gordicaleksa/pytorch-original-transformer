import argparse


import torch
from torchtext.data import Example
import matplotlib.pyplot as plt
import seaborn


from models.definitions.transformer_model import Transformer
from utils.data_utils import build_datasets_and_vocabs, build_masks_and_count_tokens
from utils.constants import *


def plot_attention_heatmap(data, x, y, head_id, ax):
    seaborn.heatmap(data, xticklabels=x, yticklabels=y, square=True, vmin=0.0, vmax=1.0, cbar=False, annot=True,
                    fmt=".2f", ax=ax)
    ax.set_title(f'MHA head id = {head_id}')


def visualize_attention_helper(attention_weights, source_sentence_tokens=None, target_sentence_tokens=None, title=''):
    num_columns = 4
    num_rows = 2
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(20, 10))  # prepare the figure and axes

    assert source_sentence_tokens is not None or target_sentence_tokens is not None, \
        f'Either source or target sentence must be passed in.'

    target_sentence_tokens = source_sentence_tokens if target_sentence_tokens is None else target_sentence_tokens
    source_sentence_tokens = target_sentence_tokens if source_sentence_tokens is None else source_sentence_tokens

    for head_id, head_attention_weights in enumerate(attention_weights):
        row_index = int(head_id / num_columns)
        column_index = head_id % num_columns
        plot_attention_heatmap(head_attention_weights, source_sentence_tokens, target_sentence_tokens if head_id % num_columns == 0 else [], head_id, axs[row_index, column_index])

    fig.suptitle(title)
    plt.show()


def visualize_attention(baseline_transformer, source_sentence_tokens, target_sentence_tokens):
    encoder = baseline_transformer.encoder
    decoder = baseline_transformer.decoder

    # Remove the end of sentence token </s> as we never attend to it, it's produced at the output and we stop
    target_sentence_tokens = target_sentence_tokens[:-1]

    # Visualize encoder attention weights
    for layer_id, encoder_layer in enumerate(encoder.encoder_layers):
        mha = encoder_layer.multi_headed_attention  # Every encoder layer has 1 MHA module

        # attention_weights shape = (B, NH, S, S), extract 0th batch and loop over NH (number of heads) MHA heads
        # S stands for maximum source token-sequence length
        attention_weights = mha.attention_weights.cpu().numpy()[0]

        title = f'Encoder layer {layer_id + 1}'
        visualize_attention_helper(attention_weights, source_sentence_tokens, title=title)

    # Visualize decoder attention weights
    for layer_id, decoder_layer in enumerate(decoder.decoder_layers):
        mha_trg = decoder_layer.trg_multi_headed_attention  # Extract the self-attention MHA
        mha_src = decoder_layer.src_multi_headed_attention  # Extract the source attending MHA

        # attention_weights shape = (B, NH, T, T), T stands for maximum target token-sequence length
        attention_weights_trg = mha_trg.attention_weights.cpu().numpy()[0]
        # shape = (B, NH, T, S), target token representations create queries and keys/values come from the encoder
        attention_weights_src = mha_src.attention_weights.cpu().numpy()[0]

        title = f'Decoder layer {layer_id + 1}, self-attention MHA'
        visualize_attention_helper(attention_weights_trg, target_sentence_tokens=target_sentence_tokens, title=title)

        title = f'Decoder layer {layer_id + 1}, source-attending MHA'
        visualize_attention_helper(attention_weights_src, source_sentence_tokens, target_sentence_tokens, title)


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

    # Step 3: Prepare the input sentence and output prompt
    source_sentence = translation_config['german_sentence']
    ex = Example.fromlist([source_sentence], fields=[('src', src_field_processor)])  # tokenize the sentence

    source_sentence_tokens = ex.src
    print(f'Source sentence tokens = {source_sentence_tokens}')

    # Numericalize and convert to cuda tensor
    src_token_ids_batch = src_field_processor.process([source_sentence_tokens], device)

    target_sentence_tokens = [BOS_TOKEN]  # initial prompt - beginning/start of the sentence token
    trg_token_ids_batch = torch.tensor([[trg_field_processor.vocab.stoi[token] for token in target_sentence_tokens]], device=device)

    # Decoding could be further optimized to cache old token activations because they can't look ahead and so
    # adding a newly predicted token won't change old token's activations.
    #
    # Example: we input <s> and do a forward pass. We get intermediate activations for <s> and at the output at position
    # 0, after the doing linear layer we get e.g. token <I>. Now we input <s>,<I> but <s>'s activations will remain
    # the same. Similarly say we now got <am> at output position 1, in the next step we input <s>,<I>,<am> and so <I>'s
    # activations will remain the same as it only looks at/attends to itself and to <s> and so forth.

    with torch.no_grad():
        # Step 4: Optimization - compute the source token representations only once
        src_mask, _, _, _ = build_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch, pad_token_id, device)
        src_representations_batch = baseline_transformer.encode(src_token_ids_batch, src_mask)

        # Step 5: Decoding process
        while True:
            _, trg_mask, _, _ = build_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch, pad_token_id, device)
            predicted_log_distributions = baseline_transformer.decode(trg_token_ids_batch, src_representations_batch, trg_mask, src_mask)

            # todo: add beam decoding mechanism
            # Greedy decoding
            most_probable_word_index = torch.argmax(predicted_log_distributions[-1]).cpu().numpy()
            predicted_word = trg_field_processor.vocab.itos[most_probable_word_index]

            if predicted_word == EOS_TOKEN:
                target_sentence_tokens.append(EOS_TOKEN)
                break

            target_sentence_tokens.append(predicted_word)
            trg_token_ids_batch = torch.tensor([[trg_field_processor.vocab.stoi[token] for token in target_sentence_tokens]], device=device)

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
