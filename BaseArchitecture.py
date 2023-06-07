# this Module is the Transformer architecture for now
import torch
import math
import torch.nn as nn
import torch.nn.functional as func
import numpy as np


# class for Front-End + Patches
class FrontEnd(nn.Module):

    def __init__(self, inputdim=400, output_dim=64, latent_dim=2048, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features=inputdim, out_features=latent_dim)
        self.fc2 = nn.Linear(in_features=latent_dim, out_features=output_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.init_weights()
        # print(self.fc1.weight.dtype)

        # Information
        self.feed_forward_param_num = inputdim * output_dim * latent_dim

    def init_weights(self):
        initrange = 0.1
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, x):
        # print(x.dtype)
        x = self.dropout(func.relu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def transformer_block(dim_model, num_head, dim_feedforward, dropout,
                      num_encode_layers, dense_dim, pooling_kernel, stride):
    encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model,
                                               nhead=num_head,
                                               dim_feedforward=dim_feedforward,
                                               dropout=dropout,
                                               batch_first=True,
                                               norm_first=True)
    sequence = nn.Sequential(
        nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_encode_layers),
        nn.Linear(in_features=dim_model, out_features=dense_dim),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=dense_dim, out_features=dense_dim),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=dense_dim, out_features=dim_model),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        LambdaLayer(lambda x: x.transpose(1, 2)),
        nn.AvgPool1d(kernel_size=pooling_kernel, stride=stride),
        LambdaLayer(lambda x: x.transpose(1, 2)))
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
            dense_dim=128,
            num_head=8,
            pooling_kernel_initial=2,
            pooling_kernel_last=10,
            num_encode_layers=2,
            dim_feedforward=2048,
            label_number=200
    ):
        super().__init__()

        # Information
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # Layers
        self.positional_encoder = PositionalEncoding(d_model=dim_model, dropout=dropout)
        self.embedding = FrontEnd(inputdim=400, output_dim=dim_model, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.output1 = nn.Linear(in_features=dim_model, out_features=dim_feedforward)
        self.output2 = nn.Linear(in_features=dim_feedforward, out_features=label_number)
        self.transformer_1 = transformer_block(dim_model=dim_model, num_head=num_head, dim_feedforward=dim_feedforward,
                                               dropout=dropout, num_encode_layers=num_encode_layers,
                                               dense_dim=dense_dim, pooling_kernel=pooling_kernel_initial, stride=2)
        self.transformer_2 = transformer_block(dim_model=dim_model, num_head=num_head, dim_feedforward=dim_feedforward,
                                               dropout=dropout, num_encode_layers=num_encode_layers,
                                               dense_dim=dense_dim, pooling_kernel=pooling_kernel_initial, stride=2)
        self.transformer_3 = transformer_block(dim_model=dim_model, num_head=num_head, dim_feedforward=dim_feedforward,
                                               dropout=dropout, num_encode_layers=num_encode_layers,
                                               dense_dim=dense_dim, pooling_kernel=pooling_kernel_last, stride=1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        # initializing weights for the transformer blocks
        transformer_blocks = [self.transformer_1, self.transformer_2, self.transformer_3]
        for block in transformer_blocks:
            if isinstance(block, nn.Linear):
                block.weight.data.uniform_(initrange, initrange)
                block.bias.data.zero_()

        # initializing weights fc layers
        # linear_layers = [self.latent1, self.latent2, self.output1, self.output2]
        linear_layers = [self.output1, self.output2]
        for layer in linear_layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

    def forward(self, source):
        # source size = (Sequence Len, Batch Size)

        # Embedding phase
        source = self.embedding(source)

        # Positional Encoding
        source = self.positional_encoder(source)

        # blocks of transformer
        x1 = self.transformer_1(source)
        x1 = self.transformer_2(x1)
        x1 = self.transformer_3(x1)

        # output
        output = self.dropout(func.relu(self.output1(x1)))
        output = torch.sigmoid(self.output2(output))
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

"""

def pre_emp(y):
    y_emp = effects.preemphasis(y)
    return y_emp


# noinspection PyUnboundLocalVariable
class ClassicalFeatures:

    # should this actually be implemented as a class for now?
    # there's something not right with my implementation

    def __init__(self, sr=16000, extract_mfcc=True, extract_stft=True,
                 extract_delta=True, extract_second_delta=True,
                 pre_emphasis=True, window_size=25, window_type="hamming"):  # what inside here?
        self.sr = sr
        self.extract_mfcc = extract_mfcc
        self.extract_stft = extract_stft
        self.extract_delta = extract_delta | self.extract_mfcc
        self.extract_second_delta = extract_second_delta | self.extract_mfcc
        self.pre_emphasis = pre_emphasis
        self.window_size = self.sr / window_size
        self.window_type = window_type
        self.num_coeff = self.num_features(model_dim=64)

    def normalize(self, mat):
        
        need to understand function for normlalizing
        :return:
        

    def num_features(self, model_dim=64):
        num_of_feature_types = sum([getattr(self, f"extract_{feature_type}") for feature_type in
                                    ["mfcc", "stft", "delta", "second_delta"]])
        return math.floor(model_dim/num_of_feature_types)

    def calculate_feature_matrix(self, y):
        features = []
        if self.pre_emphasis:
            y = pre_emp(y)

        if self.extract_mfcc:
            mfcc_feat = feat.mfcc(y=y, sr=self.sr, win_length=self.window_size,
                                  n_mfcc=self.num_coeff, window=self.window_type)
            features.append(mfcc_feat)

        if self.extract_delta:
            mfcc_delta_feat = feat.delta(mfcc_feat)
            features.append(mfcc_feat)

        if self.extract_second_delta:
            mfcc_delta_2_feat = feat.delta(mfcc_delta_feat)
            features.append(mfcc_delta_2_feat)

        if self.extract_stft:
            stft_feat = feat.chroma_stft(y=y, sr=self.sr, win_length=self.window_size,
                                         window=self.window_type, n_chroma=self.num_coeff)
            features.append(mfcc_delta_2_feat)

        features_tensor = torch.tensor(np.concatenate(features, axis=0))
        # TODO: check dimensionality, add normalizing, add features
        return features_tensor
        
"""