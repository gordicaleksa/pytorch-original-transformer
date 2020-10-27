import matplotlib.pyplot as plt
import numpy as np


from transformer_model import PositionalEncoding
from optimizers_and_loss_fn import CustomLearningRateAdamOptimizer


# def visualize_custom_lr_adam():
    # c = CustomLearningRateAdamOptimizer()


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
    visualize_positional_encodings()