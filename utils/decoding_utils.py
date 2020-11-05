import enum


import torch
import numpy as np


from .constants import *
from utils.data_utils import get_masks_and_count_tokens_trg


class DecodingMethod(enum.Enum):
    GREEDY = 0,
    BEAM = 1


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
        trg_mask, _ = get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id)
        # Shape = (B*T, V) where T is the current token-sequence length and V target vocab size
        predicted_log_distributions = baseline_transformer.decode(trg_token_ids_batch, src_representations_batch, trg_mask, src_mask)

        # Extract only the indices of last token for every target sentence (we take every T-th token)
        num_of_trg_tokens = len(target_sentences_tokens[0])
        predicted_log_distributions = predicted_log_distributions[num_of_trg_tokens-1::num_of_trg_tokens]

        # This is the "greedy" part of the greedy decoding:
        # We find indices of the highest probability target tokens and discard every other possibility
        most_probable_last_token_indices = torch.argmax(predicted_log_distributions, dim=-1).cpu().numpy()

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

    # Post process the sentences - remove everything after the EOS token
    target_sentences_tokens_post = []
    for target_sentence_tokens in target_sentences_tokens:
        try:
            target_index = target_sentence_tokens.index(EOS_TOKEN) + 1
        except:
            target_index = None

        target_sentence_tokens = target_sentence_tokens[:target_index]
        target_sentences_tokens_post.append(target_sentence_tokens)

    return target_sentences_tokens_post


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

        # Initial prompt is the beginning/start of the sentence token. Make it compatible shape with source batch => (B,1)
        batch_size, S, model_dimension = src_representations_batch.shape
        target_sentences_tokens = [[BOS_TOKEN] for _ in range(batch_size)]
        trg_token_ids_batch = torch.tensor([[trg_field_processor.vocab.stoi[tokens[0]]] for tokens in target_sentences_tokens], device=device)

        # Repeat so that source sentence representations are repeated contiguously, say we have [s1, s2] we want
        # [s1, s1, s2, s2] and not [s1, s2, s1, s2] where s1 is single sentence representation
        src_representations_batch = src_representations_batch.repeat(1, beam_size, 1).view(beam_size*batch_size, -1, model_dimension)
        trg_token_ids_batch = trg_token_ids_batch.repeat(beam_size, 1)

        hypothesis_log_probs = torch.zeros((batch_size * beam_size, 1), device=device)

        while True:
            trg_mask, _ = get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id)
            # Shape = (B*BS*T, V) T - current token-sequence length, V - target vocab size, BS - beam size, B - batch
            predicted_log_distributions = baseline_transformer.decode(trg_token_ids_batch, src_representations_batch, trg_mask, src_mask)

            # Extract only the indices of last token for every target sentence (we take every T-th token)
            num_of_trg_tokens = len(target_sentences_tokens[0])
            predicted_log_distributions = predicted_log_distributions[num_of_trg_tokens - 1::num_of_trg_tokens]

            # This time extract beam_size number of highest probability tokens (compare to greedy's arg max)
            latest_token_log_probs, most_probable_token_indices = torch.topk(predicted_log_distributions, beam_size, dim=-1, sorted=True)

            # Calculate probabilities for every beam hypothesis (since we have log prob we add instead of multiply)
            # Shape = (BS*B, BS)
            hypothesis_pool_log_probs = hypothesis_log_probs + latest_token_log_probs
            hypothesis_pool_log_probs = hypothesis_pool_log_probs.view(batch_size, beam_size, beam_size)

            # Figure out indices of beam_size most probably hypothesis
            _, new_indices = torch.topk(hypothesis_pool_log_probs.flatten(), beam_size)

            i = np.array(np.unravel_index(new_indices.cpu().numpy(), hypothesis_pool_log_probs.cpu().numpy().shape)).T

            indices_np = most_probable_token_indices.cpu().numpy()
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

