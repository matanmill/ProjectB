# this Module is the Transformer architecture for now
import torch
import math
import torch.nn as nn
from torch.nn import Transformer


# defining a class for transformer
class Transformer(nn.Module):
    """
    Initial Transformer class (16/12/22), from a guide to transformer architectures
    with some minor changes.
    """
    # constructor
    def __init__(
            self,
            dim_model,
            num_head,
            num_encode_layers,
            num_decode_layers,
            dim_feedforward,
            activation,
            dropout,
            norm_first
    ):
        super().__init__()

        # Information
        self.model_type = "Transformer"
        self.dim_model = dim_model


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
        divisoin_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)

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


