"""
    Contains the implementation of the original transformer paper "Attention is all you need".

    Paper link: https://arxiv.org/pdf/1706.03762.pdf

    Certain modifications:
    1. LayerNorm (before instead of after)
    2. Dropout (Added additionally to attention weights and point-wise feed-forward net sublayer

    Suggested theory: https://jalammar.github.io/illustrated-transformer/ (amazing blog!)

"""


import math
import copy


import torch
import torch.nn as nn


from constants import *


# todo: after I get the training up and running verify that this design is the best one
class Transformer(nn.Module):

    def __init__(self, model_dimension, src_vocab_size, tgt_vocab_size, number_of_heads, number_of_layers, dropout_probability):
        super().__init__()

        self.src_embedding = Embedding(src_vocab_size, model_dimension)
        self.src_pos_embedding = PositionalEncoding(model_dimension, dropout_probability)

        self.tgt_embedding = Embedding(tgt_vocab_size, model_dimension)
        self.tgt_pos_embedding = PositionalEncoding(model_dimension, dropout_probability)

        # All of these will get deep-copied internally
        mha = MultiHeadedAttention(model_dimension, number_of_heads, dropout_probability)
        pwn = PositionwiseFeedForwardNet(model_dimension, dropout_probability)
        encoder_layer = EncoderLayer(model_dimension, dropout_probability, mha, pwn)
        decoder_layer = DecoderLayer(model_dimension, dropout_probability, mha, pwn)

        self.encoder = Encoder(encoder_layer, number_of_layers)
        self.decoder = Decoder(decoder_layer, number_of_layers)

        self.decoder_generator = DecoderGenerator(model_dimension, tgt_vocab_size)
        self.init_params()

    def init_params(self):
        # todo: potentially add special initialization (not mentioned in the paper though)
        print('dummy')

    def forward(self, src_token_ids_batch, tgt_token_ids_batch, src_mask, tgt_mask):
        # todo: comment everything once I finished the initial design
        src_embeddings_batch = self.src_pos_embedding(self.src_embedding(src_token_ids_batch))
        src_representations_batch = self.encoder(src_embeddings_batch, src_mask)

        tgt_embeddings_batch = self.tgt_pos_embedding(self.tgt_embedding(tgt_token_ids_batch))
        tgt_representations_batch = self.decoder(tgt_embeddings_batch, src_representations_batch, tgt_mask, src_mask)

        tgt_log_probs = self.decoder_generator(tgt_representations_batch)

        return tgt_log_probs  # the reason I use log here is that PyTorch's nn.KLDivLoss expects log probabilities


#
# Encoder architecture
#


class Encoder(nn.Module):

    def __init__(self, encoder_layer, number_of_layers):
        super().__init__()
        assert isinstance(encoder_layer, EncoderLayer), f'Expected EncoderLayer got {type(encoder_layer)}.'

        self.encoder_layers = get_clones(encoder_layer, number_of_layers)
        self.norm = nn.LayerNorm(encoder_layer.model_dimension)

    def forward(self, src_embeddings_batch, src_mask):
        # Just update the naming so as to reflect the semantics of what this var will become
        src_representations_batch = src_embeddings_batch

        for encoder_layer in self.encoder_layers:
            # src_mask's role is to mask/ignore padded token representations in the multi-headed self-attention module
            src_representations_batch = encoder_layer(src_representations_batch, src_mask)

        # not mentioned explicitly in the paper
        return self.norm(src_representations_batch)


class EncoderLayer(nn.Module):

    def __init__(self, model_dimension, dropout_probability, multi_headed_attention, pointwise_net):
        super().__init__()
        num_of_sublayers_encoder = 2
        self.sublayers = get_clones(SublayerLogic(model_dimension, dropout_probability), num_of_sublayers_encoder)

        self.multi_headed_attention = multi_headed_attention
        self.pointwise_net = pointwise_net

        self.model_dimension = model_dimension

    def forward(self, src_representations_batch, src_mask):
        # Define anonymous (lambda) function which only takes src_representations_batch (srb) as input,
        # this way we have a uniform interface for the sublayer logic.
        encoder_self_attention = lambda srb: self.multi_headed_attention(query=srb, key=srb, value=srb, mask=src_mask)

        src_representations_batch = self.sublayers[0](src_representations_batch, encoder_self_attention)
        src_representations_batch = self.sublayers[1](src_representations_batch, self.pointwise_net)

        return src_representations_batch


#
# Decoder architecture
#


class Decoder(nn.Module):

    def __init__(self, decoder_layer, number_of_layers):
        super().__init__()
        assert isinstance(decoder_layer, DecoderLayer), f'Expected DecoderLayer got {type(decoder_layer)}.'

        self.decoder_layers = get_clones(decoder_layer, number_of_layers)
        self.norm = nn.LayerNorm(decoder_layer.model_dimension)

    def forward(self, tgt_embeddings_batch, src_representations_batch, tgt_mask, src_mask):
        # Just update the naming so as to reflect the semantics of what this var will become
        tgt_representations_batch = tgt_embeddings_batch

        for decoder_layer in self.decoder_layers:
            tgt_representations_batch = decoder_layer(tgt_representations_batch, src_representations_batch, tgt_mask, src_mask)

        # not mentioned explicitly in the paper
        return self.norm(tgt_representations_batch)


class DecoderLayer(nn.Module):

    def __init__(self, model_dimension, dropout_probability, multi_headed_attention, pointwise_net):
        super().__init__()
        num_of_sublayers_decoder = 3
        self.sublayers = get_clones(SublayerLogic(model_dimension, dropout_probability), num_of_sublayers_decoder)

        self.tgt_multi_headed_attention = copy.deepcopy(multi_headed_attention)
        self.src_multi_headed_attention = copy.deepcopy(multi_headed_attention)
        self.pointwise_net = pointwise_net

        self.model_dimension = model_dimension

    def forward(self, tgt_representations_batch, src_representations_batch, tgt_mask, src_mask):
        # Define anonymous (lambda) function which only takes tgt_representations_batch (trb - funny name I know)
        # as input - this way we have a uniform interface for the sublayer logic.
        srb = src_representations_batch
        decoder_tgt_self_attention = lambda trb: self.tgt_multi_headed_attention(query=trb, key=trb, value=trb, mask=tgt_mask)
        decoder_src_self_attention = lambda trb: self.src_multi_headed_attention(query=trb, key=srb, value=srb, mask=src_mask)

        tgt_representations_batch = self.sublayers[0](tgt_representations_batch, decoder_tgt_self_attention)
        tgt_representations_batch = self.sublayers[1](tgt_representations_batch, decoder_src_self_attention)
        tgt_representations_batch = self.sublayers[2](tgt_representations_batch, self.pointwise_net)

        return tgt_representations_batch


#
# Helper modules (designed with modularity in mind) and organized top to bottom.
#


class DecoderGenerator(nn.Module):
    def __init__(self, model_dimension, vocab_size):
        super().__init__()

        self.linear = nn.Linear(model_dimension, vocab_size)

        # -1 stands for apply the log-softmax along the last dimension (final token representation dimension)
        # again using log softmax as PyTorch's nn.KLDivLoss expects log probabilities (just a technical detail)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, tgt_representations_batch):
        return self.log_softmax(self.linear(tgt_representations_batch))


# Note: the original paper had LayerNorm AFTER the residual connection and addition operation
# multiple experiments I found showed that it's more effective to do it BEFORE
class SublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)

    def forward(self, representations_batch, sublayer_module):
        # Page 7, Chapter 5.4 "Regularization"
        return representations_batch + self.dropout(sublayer_module(self.norm(representations_batch)))


class PositionwiseFeedForwardNet(nn.Module):
    """
        It's position-wise because this feed forward net will be independently applied to every token's representation.

        Representations batch is of the shape (batch size, max token sequence length, model dimension).
        This net will basically be applied independently to every token's representation (you can think of it as if
        there was a nested for-loop going over the batch size and max token sequence length dimensions
        and applied this net to token representations. PyTorch does this automagically behind the scenes.

    """
    def __init__(self, model_dimension, dropout_probability, width_mult=4):
        super().__init__()

        self.linear1 = nn.Linear(model_dimension, width_mult * model_dimension)
        self.linear2 = nn.Linear(width_mult * model_dimension, model_dimension)

        # This dropout layer is not explicitly mentioned in the paper but it's common to use to avoid over-fitting
        self.dropout = nn.Dropout(p=dropout_probability)
        self.relu = nn.ReLU()

    def forward(self, representations_batch):
        return self.linear2(self.dropout(self.relu(self.linear1(representations_batch))))


class MultiHeadedAttention(nn.Module):
    """
        This module already exists in PyTorch. The reason I implemented it here from scratch is that
        PyTorch implementation is super complicated as they made it as generic as possible whereas on the other hand
        I only want to support a limited use-case.

        Also this is arguable the most important architectural component in the Transformer model.

        Additional note:
        This is conceptually super easy stuff. It's just that matrix implementation makes things a bit less intuitive.
        If you take your time and go through the code and figure out all of the dimensions + write stuff down on paper
        you'll understand everything. Also do check out this amazing blog for conceptual understanding:

        https://jalammar.github.io/illustrated-transformer/

        Optimization notes:
        qkv_nets could be replaced by Parameter(torch.empty(3 * model_dimension, model_dimension)) and one more matrix
        for bias, which would make the implementation a bit more optimized. For the sake of easier understanding though,
        I'm doing it like this - using 3 feed forward nets. Conceptually both implementations are the same.

        PyTorch's query/key/value are of different shape namely (max token sequence length, batch size, model dimension)
        whereas I'm using (batch size, max token sequence length, model dimension) because it's easier to understand
        and consistent with computer vision apps (batch dimension is always first followed by the number of channels (C)
        and image's spatial dimensions height (H) and width (W) -> (B, C, H, W).

        This has an important optimization implication, they can reshape their matrix into (B*NH, S, HD)
        (where B - batch size, S - max sequence length, NH - number of heads, HD - head dimension) in a single step
        and I can only get to (B, NH, S, HD) in single step (I could call contiguous() followed by view but that's
        expensive as it would incur additional matrix copy)

    """

    def __init__(self, model_dimension, number_of_heads, dropout_probability):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'

        self.head_dimension = int(model_dimension / number_of_heads)
        self.number_of_heads = number_of_heads

        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)  # no pun intended.
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the softmax along the last dimension

        self.attention_weights = None  # for visualization purposes

    def attention(self, query, key, value, mask):
        # Step 1: Scaled dot-product attention, Page 4, Chapter 3.2.1 "Scaled Dot-Product Attention"
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)

        # Step 2: Optionally mask words whose representations we want to ignore by setting a big negative number
        # to locations corresponding to those words (make softmax output 0 probability on those locations).
        if mask is not None:
            # todo: check whether my mask will be boolean
            scores.masked_fill(mask == 0., -float("Inf"))

        # Step 3: Calculate the attention weights - how much should we attend to surrounding token representations
        attention_weights = self.softmax(scores)

        # Step 4: Not defined in the original paper apply dropout to attention weights as well
        attention_weights = self.attention_dropout(attention_weights)

        # Step 5: based on attention weights calculate new token representations
        intermediate_token_representations = torch.matmul(attention_weights, value)

        # todo: visualize attention
        return intermediate_token_representations, attention_weights  # pass attention weights for visualization purposes

    def forward(self, query, key, value, mask):
        # Step 1: linearly project
        # todo: verify q/k/v are contiguous, try directly reshaping into PyTorch's optimal shape
        batch_size = query.shape[0]
        query, key, value = [net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
                             for net, x in zip(self.qkv_nets, (query, key, value))]

        # Step 2: Apply attention
        intermediate_token_representations, self.attention_weights = self.attention(query, key, value, mask)

        # Step 3: reshape from (B, NH, S, HD) over (B, S, NH, HD) (via transpose) into (B, S, NHxHD)
        # where B - batch size, S - max sequence length, NH - number of heads, HD - head dimension
        reshaped = intermediate_token_representations.transpose(1, 2).contiguous().view(batch_size, -1, self.number_of_heads * self.head_dimension)

        # Step 4: Linear projection
        token_representations = self.out_projection_net(reshaped)

        return token_representations


#
# Input modules
#


class Embedding(nn.Module):

    def __init__(self, vocab_size, model_dimension):
        super().__init__()
        self.embeddings_table = nn.Embedding(vocab_size, model_dimension)
        self.model_dimension = model_dimension

    def forward(self, token_ids_batch):
        assert token_ids_batch.ndim == 2, \
            f'Expected: (batch size, max token sequence length), got {token_ids_batch.shape}'

        # (stated in the paper) multiply the embedding weights by the square root of model dimension
        # Page 5, Chapter 3.4 "Embeddings and Softmax"
        return self.embeddings_table(token_ids_batch) * math.sqrt(self.model_dimension)


class PositionalEncoding(nn.Module):

    def __init__(self, model_dimension, dropout_probability, expected_max_sequence_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)

        # (stated in the paper) Use sine functions whose frequencies form a geometric progression as position encodings,
        # (learning encodings will also work!). Page 6, Chapter 3.5 "Positional Encoding"
        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension)

        # Checkout playground.py for visualization of how these look like (it's super simple don't get scared)
        positional_encodings_table = torch.zeros(expected_max_sequence_length, model_dimension)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies)  # sine on even positions
        positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies)  # cosine on odd positions
        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

        # (stated in the paper) Applying dropout to the sum of positional encodings and token embeddings
        # Page 7, Chapter 5.4 "Regularization"
        return self.dropout(embeddings_batch + self.positional_encodings_table[:embeddings_batch.shape[1]])


#
# Helper model functions
#


def get_clones(module, num_of_deep_copies):
    # Create deep copies so that we can tweak each module's weights independently
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])


# Count how many trainable weights the model has <- just for having a feeling for how big the model is
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def analyze_state_dict_shapes_and_names(model):
    # This part helped me figure out that I don't have positional encodings saved in the state dict
    print(model.state_dict().keys())

    # This part helped me see that src MHA was missing in the decoder since both it and tgt MHA were referencing
    # the same MHA object in memory - stupid mistake, happens all the time, embrace the suck!
    for name, param in model.named_parameters():
        print(name, param.shape)
        if not param.requires_grad:
            raise Exception('Expected all of the params to be trainable - no param freezing used.')


# Testing the correctness of the transformer model - feel free to ignore - I used it during model development
if __name__ == "__main__":
    use_big_transformer = True

    # Dummy data
    src_vocab_size = 11
    tgt_vocab_size = 11
    src_token_ids_batch = torch.randint(1, 10, size=(3, 2))
    tgt_token_ids_batch = torch.randint(1, 10, size=(3, 2))

    transformer = Transformer(
        model_dimension=BIG_MODEL_DIMENSION if use_big_transformer else BASELINE_MODEL_DIMENSION,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        number_of_heads=BIG_MODEL_NUMBER_OF_HEADS if use_big_transformer else BASELINE_MODEL_NUMBER_OF_HEADS,
        number_of_layers=BIG_MODEL_NUMBER_OF_LAYERS if use_big_transformer else BASELINE_MODEL_NUMBER_OF_LAYERS,
        dropout_probability=BIG_MODEL_DROPOUT_PROB if use_big_transformer else BASELINE_MODEL_DROPOUT_PROB
    )

    # These 2 functions helped me figure out the 2 bugs I had:
    # 1) I did not register positional encodings and thus they wouldn't be saved and later model-loading would fail
    # 2) I had a bug with MHA (attention) in decoder, where both src and tgt were referencing the same MHA object in mem
    # It's a good practice to see whether the names, shapes and number of params make sense.
    # e.g. I knew that the big transformer had ~175 M params and I verified that here.
    analyze_state_dict_shapes_and_names(transformer)
    print(f'Size of the {"big" if use_big_transformer else "baseline"} transformer = {count_parameters(transformer)}')

    out = transformer(src_token_ids_batch, tgt_token_ids_batch, src_mask=None, tgt_mask=None)
