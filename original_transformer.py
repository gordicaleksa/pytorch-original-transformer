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

"""
    Contains the implementation of the original transformer paper "Attention is all you need".

    Paper link: https://arxiv.org/pdf/1706.03762.pdf

"""
import math


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
        src_embeddings_batch = self.src_pos_embedding(self.src_embedding(src_token_ids_batch))
        src_representations_batch = self.encoder(src_embeddings_batch, src_mask)

        tgt_embeddings_batch = self.tgt_pos_embedding(self.tgt_embedding(tgt_token_ids_batch))
        tgt_representations_batch = self.decoder(tgt_embeddings_batch, src_representations_batch, tgt_mask)

        tgt_log_probs = self.decoder_generator(tgt_representations_batch)

        return tgt_log_probs  # the reason I use log here is that PyTorch's nn.KLDivLoss expects log probabilities


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
    def __init__(self, model_dimension, dropout, expected_max_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # (paper) Use sine functions whose frequencies form a geometric progression as encodings
        # (learning encodings will also work). Page 6, Chapter 3.5 "Positional Encoding"
        position_id = torch.arange(0, expected_max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dimension, 2) * -(math.log(10000.0) / model_dimension))

        self.positional_encodings_table = torch.zeros(expected_max_length, model_dimension)
        self.positional_encodings_table[:, 0::2] = torch.sin(position_id * div_term)
        self.positional_encodings_table[:, 1::2] = torch.cos(position_id * div_term)

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'

        # (stated in the paper) Applying dropout to the sum of positional encodings and token embeddings
        # Page 7, Chapter 5.4 "Regularization"
        return self.dropout(embeddings_batch + self.positional_encodings_table[:embeddings_batch.shape[1]])


if __name__ == "__main__":
    brt = torch.randint(1, 10, size=(3, 2))
    model_dim = 4
    em = Embedding(15, model_dim)
    pe = PositionalEncoding(model_dim)
    a = em(brt)
    b = pe(a)
    print(a)


