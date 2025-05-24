import torch
import torch.nn as nn


class PostProcessingNetwork(nn.Module):
    def __init__(self, max_iterations=20):
        super().__init__()
        self.s = nn.Parameter(torch.log(torch.tensor(9.0)))
        self.alpha = nn.Parameter(torch.tensor(0.01))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.gamma_alpha = nn.Parameter(torch.tensor(0.99))
        self.gamma_beta = nn.Parameter(torch.tensor(0.99))
        self.rho = nn.Parameter(torch.tensor(1.0))
        self.w = nn.Parameter(torch.tensor(1.0))
        self.T = max_iterations

    def create_constraint_matrix(self, sequence):
        L = len(sequence)
        M = torch.zeros(L, L)

        valid_pairs = {
            ("A", "U"),
            ("U", "A"),
            ("G", "C"),
            ("C", "G"),
            ("G", "U"),
            ("U", "G"),
        }

        for i in range(L):
            for j in range(L):
                if abs(i - j) >= 4 and (sequence[i], sequence[j]) in valid_pairs:
                    M[i, j] = 1.0

        return M

    def transform_T(self, A_hat, M):
        A_squared = A_hat * A_hat
        symmetric = 0.5 * (A_squared + A_squared.T)
        return symmetric * M

    def softsign(self, x, k=10):
        return 1.0 / (1.0 + torch.exp(-k * x))

    def forward(self, U, sequence):
        L = len(sequence)
        M = self.create_constraint_matrix(sequence)

        U_processed = torch.tanh(U - self.s) * U
        A_hat = torch.tanh(U - self.s) * torch.sigmoid(U)
        A = self.transform_T(A_hat, M)
        lambda_dual = self.w * torch.relu(A.sum(dim=1) - 1)

        trajectory = []

        for t in range(self.T):
            G = 0.5 * U_processed - torch.outer(
                lambda_dual * self.softsign(A.sum(dim=1) - 1), torch.ones(L)
            )

            decay_alpha = self.gamma_alpha**t
            A_dot = A_hat + self.alpha * decay_alpha * A_hat * M * (G + G.T)

            A_hat = torch.relu(torch.abs(A_dot) - self.rho * self.alpha * decay_alpha)
            A_hat = torch.clamp(A_hat, 0, 1)

            A = self.transform_T(A_hat, M)
            decay_beta = self.gamma_beta**t
            lambda_dual = lambda_dual + self.beta * decay_beta * torch.relu(
                A.sum(dim=1) - 1
            )

            trajectory.append(A.clone())

        return trajectory
