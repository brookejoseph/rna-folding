import torch
import torch.nn as nn


def differentiable_f1_loss(A_pred, A_true):
    if A_true.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)

    TP = torch.sum(A_pred * A_true)
    FP = torch.sum(A_pred * (1 - A_true))
    FN = torch.sum((1 - A_pred) * A_true)

    f1 = 2 * TP / (2 * TP + FP + FN + 1e-8)
    return -f1


def train_e2efold(model, train_loader, num_epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch_idx, (sequences, true_structures) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_loss = 0
            for seq, A_true in zip(sequences, true_structures):
                trajectory = model(seq)

                seq_loss = 0
                T = len(trajectory)
                gamma = 0.9

                for t, A_t in enumerate(trajectory):
                    weight = gamma ** (T - 1 - t)
                    seq_loss += weight * differentiable_f1_loss(A_t, A_true)

                seq_loss /= T
                batch_loss += seq_loss

            batch_loss /= len(sequences)

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += batch_loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch}, Batch {batch_idx}, Loss: {batch_loss.item():.4f}"
                )
