# 24, October, 2020
# todo: step1 - figure out how the first part works (toy example) (DONE)
# todo: step2 - rewrite the whole thing it's not a good design
# todo: step3 - add real example support (rewrite)
# todo: step4 - add beam search and other missing components
# todo: step5 - train the model and report BLEU
# todo: step6 - write README and open-source

# todo: take a look at the naming used in the original paper
# todo: use built-in blocks but implement them myself also
# todo: a must: implement simple, dedicated multi-headed SELF-attention
# todo: add a jupyter notebook

# todo: do it like this better then annotated transformer imp
# attn_output_weights = torch.bmm(q, k.transpose(1, 2))
#     assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

# todo: also view should have 3 dims like (batch_size * n_heads, num_tokens, head_dim)
# and not separate
# attn_output_weights.masked_fill_(attn_mask, float('-inf'))
#
# attn_output = torch.bmm(attn_output_weights, v)
#     assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
# attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
#     attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

# todo: create this in a similar fashion to GANs repo, things I've modified, etc.
"""
    Contains the implementation of the original transformer paper "Attention is all you need".

    Certain modifications:
    1. LayerNorm
    2. Dropout

    Paper link: https://arxiv.org/pdf/1706.03762.pdf

    Prerequisite theory: https://jalammar.github.io/illustrated-transformer/ (amazing blog!)

"""

import math
import copy


import torch
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self, src_embedding, tgt_embedding, src_pos_embedding, tgt_pos_embedding, encoder, decoder, decoder_generator):
        super().__init__()

        self.src_embedding = src_embedding
        self.src_pos_embedding = src_pos_embedding

        self.tgt_embedding = tgt_embedding
        self.tgt_pos_embedding = tgt_pos_embedding

        self.encoder = encoder
        self.decoder = decoder

        self.decoder_generator = decoder_generator

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
            # src_mask's role is to mask (ignore) the padded token representations in the multi-headed self-attention module
            src_representations_batch = encoder_layer(src_representations_batch, src_mask)

        return self.norm(src_representations_batch)


class EncoderLayer(nn.Module):

    def __init__(self, model_dimension, dropout_probability, multi_headed_attention, pointwise_net):
        super().__init__()
        num_of_sublayers_encoder = 2
        self.sublayers = get_clones(SublayerLogic(model_dimension, dropout_probability), num_of_sublayers_encoder)

        self.multi_headed_attention = multi_headed_attention
        self.pointwise_net = pointwise_net

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

        return self.norm(tgt_representations_batch)


class DecoderLayer(nn.Module):

    def __init__(self, model_dimension, dropout_probability, src_multi_headed_attention, tgt_multi_headed_attention, pointwise_net):
        super().__init__()
        num_of_sublayers_decoder = 3
        self.sublayers = get_clones(SublayerLogic(model_dimension, dropout_probability), num_of_sublayers_decoder)

        self.tgt_multi_headed_attention = tgt_multi_headed_attention
        self.src_multi_headed_attention = src_multi_headed_attention
        self.pointwise_net = pointwise_net

    def forward(self, tgt_representations_batch, src_representations_batch, tgt_mask, src_mask):
        # Define anonymous (lambda) function which only takes tgt_representations_batch (trb - funny name I know)
        # as input - this way we have a uniform interface for the sublayer logic.
        srb = src_representations_batch
        decoder_tgt_self_attention = lambda trb: self.multi_headed_attention(query=trb, key=trb, value=trb, mask=tgt_mask)
        decoder_src_self_attention = lambda trb: self.multi_headed_attention(query=trb, key=srb, value=srb, mask=src_mask)

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

    def __init__(self):
        super().__init__()

    def forward(self):
        print('dummy')


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

    # todo: register_buffer try coding models.params() once I have the full model and check whether these could become trainable
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
        self.positional_encodings_table = positional_encodings_table

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

        # (stated in the paper) Applying dropout to the sum of positional encodings and token embeddings
        # Page 7, Chapter 5.4 "Regularization"
        return self.dropout(embeddings_batch + self.positional_encodings_table[:embeddings_batch.shape[1]])


def get_clones(module, num_of_deep_copies):
    # Create deep copies so that we can tweak each module's weights independently
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])


if __name__ == "__main__":
    brt = torch.randint(1, 10, size=(3, 2))
    model_dim = 4
    em = Embedding(15, model_dim)
    pe = PositionalEncoding(model_dim, 0.1)
    a = em(brt)
    b = pe(a)
    print(a)


