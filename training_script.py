"""
    Notes:
        * I won't add model checkpoint averaging as mentioned in the paper - it just feels like an arbitrary heuristic
         and it won't add anything to the learning experience this repo aims to provide.

"""

# todo: write README (add attention visualization to README, plot signal tokens vs pad tokens for different
#  bucketiterator setup) and open-source
# todo: fix the integer division warning

import argparse
import time


import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu


from utils.optimizers_and_distributions import CustomLRAdamOptimizer, LabelSmoothingDistribution
from models.definitions.transformer_model import Transformer
from utils.data_utils import get_data_loaders, get_masks_and_count_tokens, get_src_and_trg_batches
import utils.utils as utils
from utils.constants import *
from utils.decoding_utils import greedy_decoding


# Global vars for logging purposes
num_of_trg_tokens_processed = 0
bleu_scores = []
global_train_step, global_val_step = [0, 0]
writer = SummaryWriter()  # (tensorboard) writer will output to ./runs/ directory by default


# Calculate the BLEU-4 score
def calculate_bleu_score(val_token_ids_loader):
    with torch.no_grad():
        print('todo')
        # hypothesis = ['This', 'is', 'cat']
        # reference = ['This', 'is', 'a', 'cat']
        # references = [reference]  # list of references for 1 sentence.
        # bleu_score = sentence_bleu(references, hypothesis)


# Simple decorator function so that I don't have to pass these arguments every time I call get_train_val_loop
def get_train_val_loop(baseline_transformer, custom_lr_optimizer, kl_div_loss, label_smoothing, pad_token_id, time_start):

    def train_val_loop(is_train, token_ids_loader, epoch):
        global num_of_trg_tokens_processed, global_train_step, global_val_step, writer

        if is_train:
            baseline_transformer.train()
        else:
            baseline_transformer.eval()

        device = next(baseline_transformer.parameters()).device

        #
        # Main loop - start of the CORE PART
        #
        for batch_idx, token_ids_batch in enumerate(token_ids_loader):
            src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt = get_src_and_trg_batches(token_ids_batch)
            src_mask, trg_mask, num_src_tokens, num_trg_tokens = get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch_input, pad_token_id, device)

            # log because the KL loss expects log probabilities (just an implementation detail)
            predicted_log_distributions = baseline_transformer(src_token_ids_batch, trg_token_ids_batch_input, src_mask, trg_mask)
            smooth_target_distributions = label_smoothing(trg_token_ids_batch_gt)  # these are regular probabilities

            if is_train:
                custom_lr_optimizer.zero_grad()  # clean the trainable weights gradients in the computational graph

            loss = kl_div_loss(predicted_log_distributions, smooth_target_distributions)

            if is_train:
                loss.backward()  # compute the gradients for every trainable weight in the computational graph
                custom_lr_optimizer.step()  # apply the gradients to weights

            # End of CORE PART

            #
            # Logging and metrics
            #

            # todo: add BLEU
            # todo: gradient clipping for stability?

            if is_train:
                global_train_step += 1
                num_of_trg_tokens_processed += num_trg_tokens

                if training_config['enable_tensorboard']:
                    writer.add_scalar('training_loss', loss.item(), global_train_step)

                if training_config['console_log_freq'] is not None and batch_idx % training_config['console_log_freq'] == 0:
                    print(f'Transformer training: time elapsed= {(time.time() - time_start):.2f} [s] '
                          f'| epoch={epoch + 1} | batch= {batch_idx + 1} '
                          f'| target tokens/batch= {num_of_trg_tokens_processed / training_config["console_log_freq"]}')

                    num_of_trg_tokens_processed = 0

                # Save model checkpoint
                if training_config['checkpoint_freq'] is not None and (epoch + 1) % training_config['checkpoint_freq'] == 0 and batch_idx == 0:
                    ckpt_model_name = f"transformer_ckpt_epoch_{epoch + 1}.pth"
                    torch.save(utils.get_training_state(training_config, baseline_transformer), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))
            else:
                global_val_step += 1

                if training_config['enable_tensorboard']:
                    writer.add_scalar('val_loss', loss.item(), global_val_step)

    return train_val_loop


def train_transformer(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU, I hope so!

    # Step 1: Prepare data loaders
    train_token_ids_loader, val_token_ids_loader, src_field_processor, trg_field_processor = get_data_loaders(training_config['dataset_path'], training_config['english_to_german'], training_config['batch_size'], device)

    pad_token_id = src_field_processor.vocab.stoi[PAD_TOKEN]  # pad token id is the same for target as well
    src_vocab_size = len(src_field_processor.vocab)
    trg_vocab_size = len(trg_field_processor.vocab)

    # Step 2: Prepare the model (original transformer) and push to GPU
    baseline_transformer = Transformer(
        model_dimension=BASELINE_MODEL_DIMENSION,
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        number_of_heads=BASELINE_MODEL_NUMBER_OF_HEADS,
        number_of_layers=BASELINE_MODEL_NUMBER_OF_LAYERS,
        dropout_probability=BASELINE_MODEL_DROPOUT_PROB
    ).to(device)

    # Step 3: Prepare other training related utilities
    kl_div_loss = nn.KLDivLoss(reduction='mean')

    # Makes smooth target distributions as opposed to conventional one-hot distributions
    # My feeling is that this is a really dummy and arbitrary heuristic but time will tell.
    label_smoothing = LabelSmoothingDistribution(BASELINE_MODEL_LABEL_SMOOTHING_VALUE, pad_token_id, trg_vocab_size, device)

    # Check out playground.py for an intuitive visualization of how the LR changes with time/training steps, easy stuff.
    custom_lr_optimizer = CustomLRAdamOptimizer(
                Adam(baseline_transformer.parameters(), betas=(0.9, 0.98), eps=1e-9),
                BASELINE_MODEL_DIMENSION,
                training_config['num_warmup_steps']
            )

    # The decorator function makes things cleaner since there is a lot of redundancy between the train and val loops
    train_val_loop = get_train_val_loop(baseline_transformer, custom_lr_optimizer, kl_div_loss, label_smoothing, pad_token_id, time.time())

    # Step 4: Start the training
    for epoch in range(training_config['num_of_epochs']):
        # Training loop
        train_val_loop(is_train=True, token_ids_loader=train_token_ids_loader, epoch=epoch)

        # Validation loop
        with torch.no_grad():
            train_val_loop(is_train=False, token_ids_loader=val_token_ids_loader, epoch=epoch)

    calculate_bleu_score(val_token_ids_loader)

    # Save the latest transformer in the binaries directory
    torch.save(utils.get_training_state(training_config, baseline_transformer), os.path.join(BINARIES_PATH, utils.get_available_binary_name()))


if __name__ == "__main__":
    #
    # Fixed args - don't change these unless you have a good reason
    #
    num_warmup_steps = 4000

    #
    # Modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=5)
    # You should adjust this for your particular machine (I have RTX 2080 with 8 GBs of VRAM so this fits nicely!)
    parser.add_argument("--batch_size", type=int, help="target number of tokens in a src/trg batch", default=1500)
    parser.add_argument("--dataset_path", type=str, help='save dataset to this path', default=os.path.join(os.path.dirname(__file__), '.data'))
    parser.add_argument("--english_to_german", type=bool, help="train the English to German model or vice versa", default=True)

    # Logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", type=bool, help="enable tensorboard logging", default=True)
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq", default=10)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq", default=1)
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    training_config['num_warmup_steps'] = num_warmup_steps

    # Train the original transformer model
    train_transformer(training_config)



