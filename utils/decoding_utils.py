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
