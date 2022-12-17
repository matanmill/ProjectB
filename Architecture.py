# this Module is the Transformer architecture for now
import torch
import math
import torch.nn as nn
import torch.nn.functional as func

######################################### IMPORTNAT NOTES #########################################
# 1. Embedding layer needs to be adjusted to main article form - I don't know how to right now
# Article that is the inspiration for embedding costs money - talk with Yael
# 2. many hyperparameters are left out - we should review what is important
# 3. target in forward of transformer class is actually the Query matrix, maybe we don't need it
# 4. Need to add - Making, multiscale Embedding, training, validation, fixing mistakes
######################################### IMPORTNAT NOTES #########################################


# class for Front-End + Patches
class FrontEnd (nn.Module):

    def __init__(self, inputdim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features=inputdim, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=output_dim)

        # Information
        self.feed_forward_param_num =inputdim*output_dim*2048

    def forward(self,x):
        x = self.fc1(x)
        x = func.relu(self.fc2(x))
        return x



# defining a class for transformer
class Transformer(nn.Module):
    """
    Initial Transformer class (16/12/22), from a guide to transformer architectures
    with some minor changes.
    """
    # constructor
    def __init__(
            self,
            dim_model: int,
            num_head: int,
            num_encode_layers: int,
            num_decode_layers: int,
            dim_feedforward: int,
            activation: str,
            dropout: float,
            norm_first: bool,
            batch_first: bool,
            max_len_tbd_1: int, # max length of positional encoding
            embedding_size_tbd_2: int,  # max length of positional encoding
            dense_dim: int # embedding_size_tbd_2 should be like dense_dim (?)
    ):
        super().__init__()

        # Information
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # Layers
        self.positional_encoder = PositionalEncoding(dim_model=dim_model, dropout=dropout, max_len=max_len_tbd_1)
        self.embedding = nn.Embedding(num_embeddings=embedding_size_tbd_2, embedding_dim=dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_head,
            num_encoder_layers=num_encode_layers,
            num_decoder_layers=num_decode_layers,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first
        )
        self.out = nn.Linear(in_features=dim_model, out_features=dense_dim)

    def forward(self,source, target, target_mask=None,source_pad_mask=None, target_pad_mask=None):

        # source size = (batch_size, source sequence length)
        # target size = (batch_size, target sequence length)

        # Embedding phase
        source = self.embedding(source) * math.sqrt(self.dim_model)
        target = self.embedding(target) * math.sqrt(self.dim_model)

        # Positional Encoding
        source = self.positional_encoder(source)
        target = self.positional_encoder(target)

        # transformer block + dense layer
        transformer_output = self.transformer(source,target,target_mask=target_mask,source_pad_mask=source_pad_mask,
                                              target_pad_mask=target_pad_mask)
        output = self.out(transformer_output)

        return output


# class for positional encoding of the transformer
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout, max_len):
        super().__init__()

        # Information
        self.dropout = nn.Dropout(dropout)
        self.dim_model = dim_model
        self.max_len = max_len

        # Encoding - Main article
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)

        # PE(pos,2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class MultiScaleEmbedding:
    # empty for now
