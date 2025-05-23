from torch import nn
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


class RNA3DFolding:
    def __init__(self, max_seq_len=512, d_model=64):
        super().__init__()
        self.d_model = d_model

        self.sequence_embedding = nn.Linear(4, d_model)
        self.pos_embedding = nn.Linear(9, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model * 3,
            nhead=2,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.conv1 = nn.Conv2d(d_model * 6, d_model, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(d_model)
        self.conv2 = nn.Conv2d(d_model, 1, kernel_size=1)

    def generate_random_weight_matrix(self, size):
        limit = torch.sqrt(torch.tensor(6.0 / (self.fan_in + self.d_model)))
        W = torch.normal(-limit, limit, size=(size, self.d_model))
        return W

    def sequence_to_matrix(self, sequence):
        current_W = self.generate_random_weight_matrix(size=4)
        matrix = []

        for base in sequence:
            matrix.append(base_mapping[base])
        final_matrix = torch.tensor(matrix, dtype=torch.float32)
        final_embedding = final_matrix @ current_W

        return final_embedding

    def position_embedding(self):
        abs_pos = torch.arange(1, self.sequence_len + 1, dtype=torch.float32)
        norm_pos = abs_pos / self.sequence_len
        pos_matrix = torch.column_stack((abs_pos, norm_pos))

        pos_embedding_list = []
        for row in pos_matrix:
            row_features = []
            row_features.append(row[0])
            row_features.append(row[1])
            row_features.append(torch.sin(row[0]))
            row_features.append(torch.sin(torch.pi * row[1]))
            row_features.append(row[1] ** 2)
            row_features.append(torch.exp(-row[1]))
            row_features.append(torch.log(1 + row[0]))
            row_features.append(1 / (1 + torch.exp(-row[1])))
            row_features.append(torch.cos(row[0]))
            pos_embedding_list.append(torch.stack(row_features))

        pos_embedding = torch.stack(pos_embedding_list)
        current_W = self.generate_random_weight_matrix(size=9)
        return pos_embedding @ current_W

    def pairwise_concat(self):
        l_x_3d_embedding = self.run_through_transformer(
            self.sequence_to_matrix(sequence)
        )
        X_i = l_x_3d_embedding.unsqueeze(1).repeat(1, self.sequence_len, 1)
        X_j = l_x_3d_embedding.unsqueeze(0).repeat(self.sequence_len, 1, 1)
        pairwise_concat = torch.cat([X_i, X_j], dim=2)
        print(pairwise_concat.shape)
        return pairwise_concat

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
        symmetric_scores = self.symmetrize_matrix_2d(output)

        return symmetric_scores

    def forward(self, sequence):
        seq_len = len(sequence)

        seq_embed = self.sequence_to_matrix(sequence)

        pos_embed = self.position_embedding(seq_len)

        combined = torch.cat([seq_embed, pos_embed], dim=1)
        transformer_out = self.transformer(combined.unsqueeze(0)).squeeze(0)

        final_embed = torch.cat([transformer_out, pos_embed], dim=1)

        scores = self.run_convolution(final_embed)

        return scores


"""

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

        
"""
