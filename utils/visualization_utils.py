import matplotlib.pyplot as plt
import seaborn


def plot_attention_heatmap(data, x, y, head_id, ax):
    seaborn.heatmap(data, xticklabels=x, yticklabels=y, square=True, vmin=0.0, vmax=1.0, cbar=False, annot=True, fmt=".2f", ax=ax)
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
    target_sentence_tokens = target_sentence_tokens[0][:-1]

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