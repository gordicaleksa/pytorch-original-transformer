import enum


import torch
import numpy as np


from .constants import *
from utils.data_utils import build_masks_and_count_tokens_trg


class DecodingMethod(enum.Enum):
    GREEDY = 0,
    BEAM = 1


# https://arxiv.org/pdf/1609.08144.pdf introduces various heuristics into the beam search algorithm like coverage
# penalty, etc. Here I only designed a simple beam search algorithm with length penalty. As the probability of the
# sequence is constructed by multiplying the conditional probabilities (which are numbers smaller than 1) the beam
# search algorithm will prefer shorter sentences which we compensate for using the length penalty.
def get_beam_decoder(translation_config):
    beam_size = translation_config['beam_size']
    length_penalty_coefficient = translation_config['length_penalty_coefficient']

    def beam_decoding(baseline_transformer, src_representations_batch, src_mask, trg_field_processor, max_target_tokens=60):
        device = next(baseline_transformer.parameters()).device
        pad_token_id = trg_field_processor.vocab.stoi[PAD_TOKEN]

        target_sentence_tokens = [BOS_TOKEN]  # initial prompt - beginning/start of the sentence token
        trg_token_ids_batch = torch.tensor([[trg_field_processor.vocab.stoi[token] for token in target_sentence_tokens]], device=device)

        hypothesis_batch = trg_token_ids_batch.repeat(beam_size, 1)
        src_representations_batch = src_representations_batch.repeat(beam_size, 1, 1)

        hypothesis_probs = torch.zeros((beam_size, 1), device=device)

        while True:
            trg_mask, _ = build_masks_and_count_tokens_trg(hypothesis_batch, pad_token_id)
            predicted_log_distributions = baseline_transformer.decode(hypothesis_batch, src_representations_batch, trg_mask, src_mask)

            log_probs, indices = torch.topk(predicted_log_distributions, beam_size, dim=-1, sorted=True)

            new_probs = hypothesis_probs + log_probs
            _, new_indices = torch.topk(new_probs.flatten(), beam_size)

            i = np.array(np.unravel_index(new_indices.cpu().numpy(), new_probs.cpu().numpy().shape)).T

            indices_np = indices.cpu().numpy()
            new_words_indices = indices_np[i]
            new_words = [trg_field_processor.vocab.itos[i] for i in new_words_indices]

            most_probable_word_index = torch.argmax(predicted_log_distributions[-1]).cpu().numpy()
            # Find the target token associated with this index
            predicted_word = trg_field_processor.vocab.itos[most_probable_word_index]

            target_sentence_tokens.append(predicted_word)
            if predicted_word == EOS_TOKEN or len(target_sentence_tokens) == max_target_tokens:
                break

            # Prepare the input for the next iteration
            trg_token_ids_batch = torch.tensor([[trg_field_processor.vocab.stoi[token] for token in target_sentence_tokens]], device=device)

        return target_sentence_tokens

    return beam_decoding


def greedy_decoding(baseline_transformer, src_representations_batch, src_mask, trg_field_processor, max_target_tokens=100):
    """
    Decoding could be further optimized to cache old token activations because they can't look ahead and so
    adding a newly predicted token won't change old token's activations.

    Example: we input <s> and do a forward pass. We get intermediate activations for <s> and at the output at position
    0, after the doing linear layer we get e.g. token <I>. Now we input <s>,<I> but <s>'s activations will remain
    the same. Similarly say we now got <am> at output position 1, in the next step we input <s>,<I>,<am> and so <I>'s
    activations will remain the same as it only looks at/attends to itself and to <s> and so forth.

    """

    device = next(baseline_transformer.parameters()).device
    pad_token_id = trg_field_processor.vocab.stoi[PAD_TOKEN]

    # Initial prompt is the beginning/start of the sentence token. Make it compatible shape with source batch => (B,1)
    target_sentences_tokens = [[BOS_TOKEN] for _ in range(src_representations_batch.shape[0])]
    trg_token_ids_batch = torch.tensor([[trg_field_processor.vocab.stoi[tokens[0]]] for tokens in target_sentences_tokens], device=device)

    # Set to true for a particular target sentence once it reaches the EOS (end-of-sentence) token
    is_decoded = [False] * src_representations_batch.shape[0]

    while True:
        trg_mask, _ = build_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id)
        predicted_log_distributions = baseline_transformer.decode(trg_token_ids_batch, src_representations_batch, trg_mask, src_mask)

        # This is the "greedy" part of the greedy decoding:
        # We find indices of the highest probability target tokens and discard every other possibility
        # Shape = (B*T, V) where T is the current max token-sequence length and V target vocab size
        most_probable_all_tokens_indices = torch.argmax(predicted_log_distributions, dim=-1).cpu().numpy()

        # Extract only the indices of last token for every target sentence (we skip every T tokens)
        num_of_trg_tokens = len(target_sentences_tokens[0])
        most_probable_last_token_indices = most_probable_all_tokens_indices[num_of_trg_tokens-1::num_of_trg_tokens]

        # Find target tokens associated with these indices
        predicted_words = [trg_field_processor.vocab.itos[index] for index in most_probable_last_token_indices]

        for idx, predicted_word in enumerate(predicted_words):
            target_sentences_tokens[idx].append(predicted_word)

            if predicted_word == EOS_TOKEN:  # once we find EOS token for a particular sentence we flag it
                is_decoded[idx] = True

        if all(is_decoded) or num_of_trg_tokens == max_target_tokens:
            break

        # Prepare the input for the next iteration (merge old token ids with the new column of most probable token ids)
        trg_token_ids_batch = torch.cat((trg_token_ids_batch, torch.unsqueeze(torch.tensor(most_probable_last_token_indices, device=device), 1)), 1)

    return target_sentences_tokens
