import torch
import torch.nn as nn
import torch.nn.functional as F


class RNA3DFolding(nn.Module):
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
        limit = torch.sqrt(torch.tensor(6.0 / (4 + self.d_model)))
        return torch.empty(size, self.d_model).uniform_(-limit.item(), limit.item())

    def sequence_to_matrix(self, sequence):
        base_mapping = {
            "A": [1, 0, 0, 0],
            "U": [0, 1, 0, 0],
            "C": [0, 0, 1, 0],
            "G": [0, 0, 0, 1],
        }
        matrix = [base_mapping[base] for base in sequence]
        final_matrix = torch.tensor(matrix, dtype=torch.float32)
        return self.sequence_embedding(final_matrix)

    def position_embedding(self, sequence_len):
        abs_pos = torch.arange(1, sequence_len + 1, dtype=torch.float32)
        norm_pos = abs_pos / (sequence_len + 1)
        pos_matrix = torch.stack([abs_pos, norm_pos], dim=1)

        pos_embedding_list = []
        for row in pos_matrix:
            features = [
                row[0],
                row[1],
                torch.sin(row[0]),
                torch.sin(torch.pi * row[1]),
                row[1] ** 2,
                torch.exp(-row[1]),
                torch.log(1 + row[0]),
                1 / (1 + torch.exp(-row[1])),
                torch.cos(row[0]),
            ]
            pos_embedding_list.append(torch.tensor(features))

        pos_embedding = torch.stack(pos_embedding_list)
        return self.pos_embedding(pos_embedding)

    def pairwise_concat(self, embedding):
        L, d = embedding.shape
        x_i = embedding.unsqueeze(1).repeat(1, L, 1)
        x_j = embedding.unsqueeze(0).repeat(L, 1, 1)
        return torch.cat([x_i, x_j], dim=-1)  # shape: L × L × 2d

    def run_convolution(self, embedding):
        pairwise = self.pairwise_concat(embedding)  # shape: L × L × 2d
        L = pairwise.shape[0]
        pairwise = pairwise.permute(2, 0, 1).unsqueeze(0)  # shape: 1 × 2d × L × L

        x = F.relu(self.bn1(self.conv1(pairwise)))
        x = self.conv2(x)  # shape: 1 × 1 × L × L
        x = x.squeeze(0).squeeze(0)  # shape: L × L
        return (x + x.T) / 2  # symmetrize

    def forward(self, sequence):
        L = len(sequence)
        seq_embed = self.sequence_to_matrix(sequence)  # L × d
        pos_embed = self.position_embedding(L)  # L × d

        combined = torch.cat([seq_embed, pos_embed], dim=1)  # L × 2d
        transformer_input = torch.cat([combined, pos_embed], dim=1).unsqueeze(
            0
        )  # L × 3d
        transformer_out = self.transformer(transformer_input).squeeze(0)  # L × 3d

        scores = self.run_convolution(transformer_out)  # L × L
        return scores
