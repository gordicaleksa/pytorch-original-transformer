import torch
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


from models.definitions.transformer_model import PositionalEncoding
from utils.optimizers_and_distributions import CustomLRAdamOptimizer, LabelSmoothingDistribution, OneHotDistribution


def display(imgs_to_display):
    num_display_imgs = 2
    assert len(imgs_to_display) == num_display_imgs, f'Expected {num_display_imgs} got {len(imgs_to_display)} images.'

    fig = plt.figure(figsize=(10, 5))
    title_fontsize = 'x-small'
    titles = ['one hot distribution', 'label smoothing distribution']

    gs = fig.add_gridspec(1, 2, left=0.02, right=0.98, wspace=0.05, hspace=0.3)

    ax = np.zeros(num_display_imgs, dtype=object)
    ax[0] = fig.add_subplot(gs[0, 0])
    ax[1] = fig.add_subplot(gs[0, 1])

    for i in range(num_display_imgs):
        ax[i].imshow(imgs_to_display[i])
        ax[i].set_title(titles[i], fontsize=title_fontsize)
        ax[i].tick_params(which='both', bottom=False, left=False, labelleft=False, labelbottom=False)

    plt.show()


def visualize_label_smoothing():
    pad_token_id = 0  # index 0 of the vocab corresponds to the pad token
    trg_vocab_size = 4  # assume only 4 words in our vocab - a toy example

    smooth = LabelSmoothingDistribution(smoothing_value=0.1, pad_token_id=pad_token_id, trg_vocab_size=trg_vocab_size)
    one_hot = OneHotDistribution(pad_token_id=pad_token_id, trg_vocab_size=trg_vocab_size)

    target = torch.tensor([[1], [2], [3], [0]])

    smooth_target_distributions = smooth(target)
    one_hot_target_distributions = one_hot(target)

    display([one_hot_target_distributions.numpy(), smooth_target_distributions.numpy()])


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
