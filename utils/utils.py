import re
import os
import time


import git
import torch
from nltk.translate.bleu_score import corpus_bleu


from .constants import BINARIES_PATH, PAD_TOKEN
from .decoding_utils import greedy_decoding
from .data_utils import get_masks_and_count_tokens_src


def get_available_binary_name():
    prefix = 'transformer'

    def valid_binary_name(binary_name):
        # First time you see raw f-string? Don't worry the only trick is to double the brackets.
        pattern = re.compile(rf'{prefix}_[0-9]{{6}}\.pth')
        return re.fullmatch(pattern, binary_name) is not None

    # Just list the existing binaries so that we don't overwrite them but write to a new one
    valid_binary_names = list(filter(valid_binary_name, os.listdir(BINARIES_PATH)))
    if len(valid_binary_names) > 0:
        last_binary_name = sorted(valid_binary_names)[-1]
        new_suffix = int(last_binary_name.split('.')[0][-6:]) + 1  # increment by 1
        return f'{prefix}_{str(new_suffix).zfill(6)}.pth'
    else:
        return f'{prefix}_000000.pth'


def get_training_state(training_config, model):
    training_state = {
        # "commit_hash": git.Repo(search_parent_directories=True).head.object.hexsha,
        "dataset_name": training_config['dataset_name'],
        "language_direction": training_config['language_direction'],

        "num_of_epochs": training_config['num_of_epochs'],
        "batch_size": training_config['batch_size'],

        "state_dict": model.state_dict()
    }

    return training_state


def print_model_metadata(training_state):
    header = f'\n{"*"*5} Model training metadata: {"*"*5}'
    print(header)

    for key, value in training_state.items():
        if key != 'state_dict':  # don't print state_dict it's a bunch of numbers...
            if key == 'language_direction':  # convert into human readable format
                value = 'English to German' if value == 'E2G' else 'German to English'
            print(f'{key}: {value}')
    print(f'{"*" * len(header)}\n')


# Calculate the BLEU-4 score
def calculate_bleu_score(transformer, token_ids_loader, trg_field_processor):
    with torch.no_grad():
        pad_token_id = trg_field_processor.vocab.stoi[PAD_TOKEN]

        gt_sentences_corpus = []
        predicted_sentences_corpus = []

        ts = time.time()
        for batch_idx, token_ids_batch in enumerate(token_ids_loader):
            src_token_ids_batch, trg_token_ids_batch = token_ids_batch.src, token_ids_batch.trg
            if batch_idx % 10 == 0:
                print(f'batch={batch_idx}, time elapsed = {time.time()-ts} seconds.')

            # Optimization - compute the source token representations only once
            src_mask, _ = get_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id)
            src_representations_batch = transformer.encode(src_token_ids_batch, src_mask)

            predicted_sentences = greedy_decoding(transformer, src_representations_batch, src_mask, trg_field_processor)
            predicted_sentences_corpus.extend(predicted_sentences)  # add them to the corpus of translations

            # Get the token and not id version of GT (ground-truth) sentences
            trg_token_ids_batch = trg_token_ids_batch.cpu().numpy()
            for target_sentence_ids in trg_token_ids_batch:
                target_sentence_tokens = [trg_field_processor.vocab.itos[id] for id in target_sentence_ids if id != pad_token_id]
                gt_sentences_corpus.append([target_sentence_tokens])  # add them to the corpus of GT translations

        bleu_score = corpus_bleu(gt_sentences_corpus, predicted_sentences_corpus)
        print(f'BLEU-4 corpus score = {bleu_score}, corpus length = {len(gt_sentences_corpus)}, time elapsed = {time.time()-ts} seconds.')
        return bleu_score
