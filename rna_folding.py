import numpy as np
from torch import nn
import torch.nn as nn
import torch

from convolution import DoubleConv2D


sequence = "GGGUGCUCAGUACGAGAGGAACCGCACCC"
sequence_len = 29
embedding_dimension = 64

base_mapping = {
    "A": [1, 0, 0, 0],
    "U": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "C": [0, 0, 0, 1],
}

nonlinear_funcs = [
    lambda x: x[0],
    lambda x: x[1],
    lambda x: np.sin(x[0]),
    lambda x: np.sin(np.pi * x[1]),
    lambda x: x[1] ** 2,
    lambda x: np.exp(-x[1]),
    lambda x: np.log(1 + x[0]),
    lambda x: 1 / (1 + np.exp(-x[1])),
    lambda x: np.cos(x[0]),
]


class RNA3DFolding:
    def __init__(self):
        self.embedding_dimension = 64
        self.sequence_len = 29
        self.fan_in = 4
        self.pos_embed = self.position_embedding()

    def generate_random_weight_matrix(self, size):
        limit = np.sqrt(6 / (self.fan_in + self.embedding_dimension))
        W = np.random.uniform(-limit, limit, size=(size, self.embedding_dimension))
        return W

    def sequence_to_matrix(self, sequence):
        current_W = self.generate_random_weight_matrix(size=4)
        matrix = []

        for base in sequence:
            matrix.append(base_mapping[base])
        final_matrix = np.array(matrix)
        final_embedding = final_matrix @ current_W

        return final_embedding

    def position_embedding(self):
        abs_pos = np.arange(1, self.sequence_len + 1)
        norm_pos = abs_pos / self.sequence_len
        pos_matrix = np.column_stack((abs_pos, norm_pos))

        pos_embedding = np.array(
            [[f(row) for f in nonlinear_funcs] for row in pos_matrix]
        )

        current_W = self.generate_random_weight_matrix(size=9)
        return pos_embedding @ current_W

    def run_through_transformer(self, embedding_pre_transformer):
        sequence_embedding = torch.tensor(
            embedding_pre_transformer, dtype=torch.float32
        )
        pos_embedding = torch.tensor(self.pos_embed, dtype=torch.float32)

        combined_embedding = torch.cat([sequence_embedding, pos_embedding], dim=1)

        combined_tensor = combined_embedding.unsqueeze(0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, batch_first=True
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        final_tensor = torch.cat(
            ([transformer_encoder(combined_tensor).squeeze(0), pos_embedding]), dim=1
        )
        return final_tensor

    def pairwise_concat(self):
        l_x_3d_embedding = self.run_through_transformer(
            self.sequence_to_matrix(sequence)
        )
        X_i = l_x_3d_embedding.unsqueeze(1).repeat(1, self.sequence_len, 1)
        X_j = l_x_3d_embedding.unsqueeze(0).repeat(self.sequence_len, 1, 1)
        pairwise_concat = torch.cat([X_i, X_j], dim=2)
        print(pairwise_concat.shape)
        return pairwise_concat

    def symmetrize_matrix(self, scores):
        scores_2d = scores.squeeze(-1)

        symmetric_scores = (scores_2d + scores_2d.T) / 2

        return symmetric_scores.unsqueeze(-1)

    def symmetrize_matrix_2d(self, scores):
        scores_2d = scores.squeeze(-1)
        return (scores_2d + scores_2d.T) / 2

    def run_convolution(self):
        pairwise_concat = self.pairwise_concat()
        _, _, six_d = pairwise_concat.shape

        pairwise_concat = pairwise_concat.unsqueeze(0)

        model = DoubleConv2D(d=int(six_d // 6))

        output = model(pairwise_concat)
        output = output.squeeze(0)
        symmetric_scores = self.symmetrize_matrix(output)

        return symmetric_scores
