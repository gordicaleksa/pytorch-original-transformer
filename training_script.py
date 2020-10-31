# todo: train the model and report BLEU
# todo: write README (add attention visualization to README) and open-source
# todo: create this in a similar fashion to GANs repo, things I've modified, etc.

import os
import argparse
import time


import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


from constants import *
from data_utils import get_data_loaders, build_masks_and_count_tokens, fetch_src_and_trg_batches
from optimizers_and_loss_fn import CustomLRAdamOptimizer, LabelSmoothingDistribution
from transformer_model import Transformer
import utils as utils


# todo: after I setup the whole train/val loop step through and make sure that attention mechanism, etc. works as expected
def train_transformer(training_config):
    writer = SummaryWriter()  # (tensorboard) writer will output to ./runs/ directory by default
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # checking whether you have a GPU

    # Step 1: Prepare data loaders and data-related information
    train_token_ids_loader, val_token_ids_loader, SRC, TRG = get_data_loaders(training_config['batch_size'], device)
    assert SRC.vocab.stoi[PAD_TOKEN] == TRG.vocab.stoi[PAD_TOKEN]

    padding_token_id = SRC.vocab.stoi[PAD_TOKEN]
    trg_vocab_size = len(TRG.vocab)

    # Step 2: Prepare the model (transformer)
    baseline_transformer = Transformer(
        model_dimension=BASELINE_MODEL_DIMENSION,
        src_vocab_size=len(SRC.vocab),
        trg_vocab_size=len(TRG.vocab),
        number_of_heads=BASELINE_MODEL_NUMBER_OF_HEADS,
        number_of_layers=BASELINE_MODEL_NUMBER_OF_LAYERS,
        dropout_probability=BASELINE_MODEL_DROPOUT_PROB
    ).to(device)

    # Step 3: Prepare other training related utilities
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')

    smoother = LabelSmoothingDistribution(BASELINE_MODEL_LABEL_SMOOTHING_VALUE, padding_token_id, trg_vocab_size, device)

    custom_lr_optimizer = CustomLRAdamOptimizer(
                Adam(baseline_transformer.parameters(), betas=(0.9, 0.98), eps=1e-9),
                BASELINE_MODEL_DIMENSION,
                training_config['num_warmup_steps']
            )

    # Step 4: Start the training
    train_loss_data = []
    val_loss_data = []
    num_of_trg_tokens_processed = 0

    ts = time.time()
    for epoch in range(training_config['num_of_epochs']):
        # Training loop
        baseline_transformer.train()
        for batch_idx, token_ids_batch in enumerate(train_token_ids_loader):
            src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt = fetch_src_and_trg_batches(token_ids_batch)
            src_mask, trg_mask, num_src_tokens, num_trg_tokens = build_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch_input, padding_token_id, device)

            # log because the KL loss expects log probabilities (just an implementation detail)
            predicted_log_distributions = baseline_transformer(src_token_ids_batch, trg_token_ids_batch_input, src_mask, trg_mask)
            target_distributions = smoother(trg_token_ids_batch_gt)

            custom_lr_optimizer.zero_grad()  # clean the gradients of every trainable weight in the computational graph
            loss = kl_div_loss(predicted_log_distributions, target_distributions)
            loss.backward()  # compute the gradients for every trainable weight in the computational graph
            custom_lr_optimizer.step()  # apply the gradients to weights

            #
            # Logging and checkpoint creation
            #

            train_loss_data.append(loss.item())
            num_of_trg_tokens_processed += num_trg_tokens

            # todo: add BLEU
            if training_config['enable_tensorboard']:
                writer.add_scalar('training_loss', loss.item(), len(train_token_ids_loader) * epoch + batch_idx + 1)

            if training_config['console_log_freq'] is not None and batch_idx % training_config['console_log_freq'] == 0:
                print(f'Transformer training: time elapsed= {(time.time() - ts):.2f} [s] '
                      f'| epoch={epoch + 1} | batch= [{batch_idx + 1}/{len(train_token_ids_loader)}] '
                      f'| target tokens/batch= {num_of_trg_tokens_processed/training_config["console_log_freq"]}')

                num_of_trg_tokens_processed = 0

            # Save model checkpoint
            if training_config['checkpoint_freq'] is not None and (epoch + 1) % training_config['checkpoint_freq'] == 0 and batch_idx == 0:
                ckpt_model_name = f"transformer_ckpt_epoch_{epoch + 1}.pth"
                torch.save(utils.get_training_state(training_config, baseline_transformer), os.path.join(CHECKPOINTS_PATH, ckpt_model_name))

        # Validation loop
        baseline_transformer.eval()
        with torch.no_grad():
            for batch_idx, token_ids_batch in enumerate(val_token_ids_loader):
                src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt = fetch_src_and_trg_batches(token_ids_batch)
                src_mask, trg_mask, num_src_tokens, num_trg_tokens = build_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch_input, padding_token_id)

                # Push to GPU (if we have a GPU on this machine)
                src_token_ids_batch = src_token_ids_batch.to(device)
                trg_token_ids_batch_input = trg_token_ids_batch_input.to(device)
                trg_token_ids_batch_gt = trg_token_ids_batch_gt.to(device)
                src_mask = src_mask.to(device)
                trg_mask = trg_mask.to(device)

                # log because the KL loss expects log probabilities (just an implementation detail)
                predicted_log_distributions = baseline_transformer(src_token_ids_batch, trg_token_ids_batch_input, src_mask, trg_mask)
                target_distributions = smoother(trg_token_ids_batch_gt)

                loss = kl_div_loss(predicted_log_distributions, target_distributions)
                val_loss_data.append(loss.item())

                if training_config['enable_tensorboard']:
                    writer.add_scalar('val_loss', loss.item(), len(val_token_ids_loader) * epoch + batch_idx + 1)

    # Save the latest generator in the binaries directory
    torch.save(utils.get_training_state(training_config, baseline_transformer), os.path.join(BINARIES_PATH, utils.get_available_binary_name()))


if __name__ == "__main__":
    #
    # fixed args - don't change these unless you have a good reason
    #
    num_warmup_steps = 4000

    #
    # modifiable args - feel free to play with these (only small subset is exposed by design to avoid cluttering)
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs", type=int, help="number of training epochs", default=5)
    parser.add_argument("--batch_size", type=int, help="number of batches", default=8)

    # logging/debugging/checkpoint related (helps a lot with experimentation)
    parser.add_argument("--enable_tensorboard", type=bool, help="enable tensorboard logging", default=True)
    parser.add_argument("--console_log_freq", type=int, help="log to output console (batch) freq", default=10)
    parser.add_argument("--checkpoint_freq", type=int, help="checkpoint model saving (epoch) freq", default=1)
    args = parser.parse_args()

    # Wrapping training configuration into a dictionary
    training_config = dict()
    for arg in vars(args):
        training_config[arg] = getattr(args, arg)
    training_config['num_warmup_steps'] = num_warmup_steps

    # train the original transformer model
    train_transformer(training_config)



