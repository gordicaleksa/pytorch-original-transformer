import torch
from torch import nn


class CustomLRAdamOptimizer:
    """
        Linear ramp learning rate for the warm-up number of steps and then start decaying
        according to the inverse square root law of the current training step number.

        Check out playground.py for visualization of the learning rate (visualize_custom_lr_adam).
    """

    def __init__(self, optimizer, model_dimension, num_of_warmup_steps):
        self.optimizer = optimizer
        self.model_size = model_dimension
        self.num_of_warmup_steps = num_of_warmup_steps

        self.current_step_number = 0

    def step(self):
        self.current_step_number += 1
        current_learning_rate = self.get_current_learning_rate()

        for p in self.optimizer.param_groups:
            p['lr'] = current_learning_rate

        self.optimizer.step()  # apply gradients

    # Check out the formula at Page 7, Chapter 5.3 "Optimizer" and playground.py for visualization
    def get_current_learning_rate(self):
        # For readability purpose
        step = self.current_step_number
        warmup = self.num_of_warmup_steps

        return self.model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))

    def zero_grad(self):
        self.optimizer.zero_grad()


class LabelSmoothingDistribution(nn.Module):
    """
        Instead of one-hot target distribution set the target word's probability to "confidence_value" (usually 0.9)
        and distribute the rest of the "smoothing_value" mass (usually 0.1) over the rest of the vocab.

        Check out playground.py for visualization of how the smooth target distribution looks like compared to one-hot.
    """

    def __init__(self, smoothing_value, padding_idx, trg_vocab_size, device):
        assert 0.0 <= smoothing_value <= 1.0

        super(LabelSmoothingDistribution, self).__init__()

        self.confidence_value = 1.0 - smoothing_value
        self.smoothing_value = smoothing_value

        self.padding_idx = padding_idx
        self.tgt_vocab_size = trg_vocab_size
        self.device = device

    def forward(self, tgt_token_ids_batch):

        batch_size = tgt_token_ids_batch.shape[0]
        smooth_target_distributions = torch.zeros((batch_size, self.tgt_vocab_size), device=self.device)

        # -2 because we are not distributing the smoothing mass over the padding index and over the ground truth index
        # those 2 values will be overwritten by the following 2 lines with confidence_value and 0 (for padding index)
        smooth_target_distributions.fill_(self.smoothing_value / (self.tgt_vocab_size - 2))

        smooth_target_distributions.scatter_(1, tgt_token_ids_batch, self.confidence_value)
        smooth_target_distributions[:, self.padding_idx] = 0.

        # If we had a padding token as a target we set the distribution to all 0s instead of smooth labeled distribution
        smooth_target_distributions.masked_fill_(tgt_token_ids_batch == self.padding_idx, 0.)

        return smooth_target_distributions


class OneHotDistribution(nn.Module):
    """
        Create a one hot distribution (feel free to ignore used only in playground.py)
    """

    def __init__(self, padding_idx, trg_vocab_size):

        super(OneHotDistribution, self).__init__()

        self.padding_idx = padding_idx
        self.trg_vocab_size = trg_vocab_size

    def forward(self, tgt_token_ids_batch):

        batch_size = tgt_token_ids_batch.shape[0]
        one_hot_distribution = torch.zeros((batch_size, self.trg_vocab_size))
        one_hot_distribution.scatter_(1, tgt_token_ids_batch, 1.)

        # If we had a padding token as a target we set the distribution to all 0s instead of one-hot distribution
        one_hot_distribution.masked_fill_(tgt_token_ids_batch == self.padding_idx, 0.)

        return one_hot_distribution
