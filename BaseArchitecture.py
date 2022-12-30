# this Module is the Transformer architecture for now
import torch
import math
import torch.nn as nn
import torch.nn.functional as func


# class for Front-End + Patches
class FrontEnd(nn.Module):

    def __init__(self, inputdim=400, output_dim=64, latent_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(in_features=inputdim, out_features=latent_dim)
        self.fc2 = nn.Linear(in_features=latent_dim, out_features=output_dim)

        self.init_weights()

        # Information
        self.feed_forward_param_num = inputdim * output_dim * latent_dim

    def init_weights(self):
        initrange = 0.1
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = func.relu(self.fc2(x))
        return x


def transformer_block(dim_model, num_head, dim_feedforward, dropout,
                      num_encode_layers, dense_dim, pooling_kernel, pooling=True):
    encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                               nhead=num_head,
                                               dim_feedforward=dim_feedforward,
                                               dropout=dropout,
                                               batch_first=True,
                                               norm_first=True),
    sequence = nn.Sequential(
        nn.TransformerEncoder(encoder_layer, num_layers=num_encode_layers),
        nn.Linear(in_features=dim_model, out_features=dense_dim),
        nn.ReLU(),
        nn.Linear(in_features=dense_dim, out_features=dense_dim),
        nn.ReLU(),
        nn.Linear(in_features=dense_dim, out_features=dense_dim),
        nn.ReLU(),
    )
    if pooling:
        sequence.append(nn.AvgPool1d(kernel_size=pooling_kernel, stride=2))
    return sequence


# defining a class for transformer
class BaseTransformer(nn.Module):
    """
    Initial Transformer class (16/12/22), from a guide to transformer architectures
    with some minor changes.
    """

    # constructor
    def __init__(
            self,
            dim_model=64,
            dropout=0.1,
            dense_dim=256,
            num_head=8,
            pooling_kernel=2,
            num_encode_layers=2,
            dim_feedforward=128,
            label_number=200
    ):
        super().__init__()

        # Information
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # Layers
        self.positional_encoder = PositionalEncoding(d_model=dim_model, dropout=dropout)
        self.embedding = FrontEnd(inputdim=400, output_dim=64)
        self.output = nn.Linear(in_features=dense_dim, out_features=label_number)
        self.transformer_1 = transformer_block(dim_model=dim_model, num_head=num_head, dim_feedforward=dim_feedforward,
                                               dropout=dropout, num_encode_layers=num_encode_layers,
                                               dense_dim=dense_dim, pooling_kernel=pooling_kernel)
        self.transformer_2 = transformer_block(dim_model=dim_model, num_head=num_head, dim_feedforward=dim_feedforward,
                                               dropout=dropout, num_encode_layers=num_encode_layers,
                                               dense_dim=dense_dim, pooling_kernel=pooling_kernel)
        self.transformer_3 = transformer_block(dim_model=dim_model, num_head=num_head, dim_feedforward=dim_feedforward,
                                               dropout=dropout, num_encode_layers=num_encode_layers,
                                               dense_dim=dense_dim, pooling_kernel=pooling_kernel, pooling=False)
        self.init_weights()  # will probably remove

    def init_weights(self):
        initrange = 0.1
        self.out.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()
        self.dense.weight.data.uniform_(-initrange, initrange)
        self.dense.bias.data.zero_()

    def forward(self, source):
        # source size = (Sequence Len, Batch Size)

        # Embedding phase
        source = self.embedding(source) * math.sqrt(self.dim_model)  # remove normalization

        # Positional Encoding
        source = self.positional_encoder(source)

        # blocks of transformer
        x1 = self.transformer_1(source)
        x1 = self.transformer_2(x1)
        x1 = self.transformer_3(x1)
        output = func.relu(self.output(x1))
        return output


class PositionalEncoding(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
