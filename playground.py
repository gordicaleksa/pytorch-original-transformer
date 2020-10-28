import torch
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


from transformer_model import PositionalEncoding
from optimizers_and_loss_fn import CustomLRAdamOptimizer, LabelSmoothing, OneHot


def visualize_label_smoothing():
    padding_idx = 0
    tgt_vocab_size = 4
    smoother = LabelSmoothing(smoothing_value=0.1, padding_idx=padding_idx, tgt_vocab_size=tgt_vocab_size)
    one_hot = OneHot(padding_idx=padding_idx, tgt_vocab_size=tgt_vocab_size)

    target = torch.tensor([[1], [2], [3], [0]])

    smooth_target_distributions = smoother(target)
    one_hot_target_distributions = one_hot(target)

    # todo: make a side by side comparision
    plt.imshow(smooth_target_distributions.numpy()); plt.show()
    print(smooth_target_distributions)


def visualize_custom_lr_adam():
    model = nn.Linear(3, 2)  # dummy model - as we need to pass some params into Adam's constructor
    num_of_warmup_steps = [2000, 4000]  # 4000 was used in the paper
    model_dimensions = [512, 1024]  # baseline and big model's dimensions

    # The base model was trained for 100k steps (Page 7, Chapter 5.2 "Hardware and Schedule"
    number_of_simulated_training_steps = 100000

    # Try a couple of different warm up steps and base/big model dimensions
    lr_data = []
    labels = []
    for warmup in num_of_warmup_steps:
        for model_dimension in model_dimensions:

            optimizer = CustomLRAdamOptimizer(
                Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9),
                model_dimension,
                warmup
            )

            label = f'warmup:{warmup}_dim:{model_dimension}'
            learning_rates = []

            # Collect the learning rates
            for i in range(number_of_simulated_training_steps):
                optimizer.step()
                learning_rates.append(optimizer.get_current_learning_rate())

            labels.append(label)
            lr_data.append(learning_rates)

    # Plot data
    num_of_steps_to_plot = 20000
    for learning_rates, label in zip(lr_data, labels):
        plt.plot(learning_rates[:num_of_steps_to_plot], label=label)
    plt.legend()
    plt.title('Learning rate schedule')
    plt.xlabel('number of training steps')
    plt.ylabel('learning rate')
    plt.show()


def visualize_positional_encodings():
    # Create a PositionalEncoding object instance
    pe = PositionalEncoding(model_dimension=512, dropout_probability=0.1)

    # Extract the positional encodings table
    positional_encodings_table = pe.positional_encodings_table.numpy()

    # Repeat columns width_mult times for better visualization
    shape = positional_encodings_table.shape
    data_type = positional_encodings_table.dtype
    width_mult = 9  # make it almost square
    positional_encodings_img = np.zeros((shape[0], width_mult*shape[1]), dtype=data_type)
    for i in range(width_mult):
        positional_encodings_img[:, i::width_mult] = positional_encodings_table

    # Display the positional encodings table
    # Every row in this table gets added to a particular token position
    # Row 0 always gets added to 0th token embedding, row 1 gets added to 1st token embedding, etc.
    plt.title('Positional encodings')
    plt.imshow(positional_encodings_img); plt.show()


if __name__ == "__main__":
    # visualize_positional_encodings()
    # visualize_custom_lr_adam()
    visualize_label_smoothing()
