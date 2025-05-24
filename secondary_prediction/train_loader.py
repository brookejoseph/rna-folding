# test_e2e.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from forward import RNA3DFolding
from backwards import PostProcessingNetwork
from train import train_e2efold, differentiable_f1_loss


class E2Efold(nn.Module):
    def __init__(self, max_seq_len=512, d_model=64, max_iterations=20):
        super().__init__()
        self.deep_score_net = RNA3DFolding(max_seq_len, d_model)
        self.post_process_net = PostProcessingNetwork(max_iterations)

    def forward(self, sequence):  # Fixed - added sequence parameter
        scores = self.deep_score_net(sequence)
        trajectory = self.post_process_net(scores, sequence)
        return trajectory


def parse_structure_to_matrix(structure_string, target_length):
    """Convert dot-bracket notation to binary matrix with specified length"""
    # Pad or truncate structure to match sequence length
    if len(structure_string) < target_length:
        structure_string += "." * (target_length - len(structure_string))
    elif len(structure_string) > target_length:
        structure_string = structure_string[:target_length]

    L = target_length
    matrix = torch.zeros(L, L)

    stack = []
    for i, char in enumerate(structure_string):
        if char == "(":
            stack.append(i)
        elif char == ")" and stack:
            j = stack.pop()
            matrix[i, j] = 1
            matrix[j, i] = 1

    return matrix


class RNADataset(Dataset):
    def __init__(self, sequences, structures):
        self.sequences = sequences
        # Pass sequence length to ensure matching dimensions
        self.structures = [
            parse_structure_to_matrix(structures[i], len(sequences[i]))
            for i in range(len(sequences))
        ]


# Replace your test_end_to_end function with this fixed version:


def custom_collate_fn(batch):
    """Custom collate function that handles variable-length sequences"""
    sequences, structures = zip(*batch)
    # Don't stack - just return lists
    return list(sequences), list(structures)


def test_end_to_end():
    print("🧬 Testing E2Efold End-to-End")

    sequences = [
        "GGGAAACGUUCCG",  # 13 bases
        "AUCGAUCGAUCGA",  # 13 bases (added A)
        "GCGCAAUUACGCG",  # 13 bases (added G)
        "UUGCGCAAGCAAG",  # 13 bases (added G)
    ]

    structures = [
        "(((....))))..",  # 13 chars
        "(((...)))....",  # 13 chars
        "((....))(...).",  # 13 chars
        "(((....))).",  # 11 chars - need: "(((....)))..""
    ]

    print(f"✅ Created dataset with {len(sequences)} samples")

    # 2. Create DataLoader with custom collate function
    dataset = RNADataset(sequences, structures)
    train_loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=custom_collate_fn,  # This fixes the error!
    )
    print(f"✅ Created DataLoader with batch_size=2")

    # 3. Initialize Model
    model = E2Efold(d_model=32, max_iterations=5)
    print(f"✅ Model initialized")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Test Forward Pass
    print("\n🔄 Testing Forward Pass...")
    test_seq = sequences[0]
    print(f"   Input sequence: {test_seq} (length: {len(test_seq)})")

    with torch.no_grad():
        trajectory = model(test_seq)
        print(f"   Output trajectory length: {len(trajectory)}")
        print(f"   Each matrix shape: {trajectory[0].shape}")
        print(f"   Final prediction sample:")
        print(f"   {trajectory[-1][:5, :5]}")

    # 5. Test Loss Computation
    print("\n📊 Testing Loss Computation...")
    A_true = dataset.structures[0]
    A_pred = trajectory[-1]

    loss = differentiable_f1_loss(A_pred, A_true)
    print(f"   F1 Loss: {loss.item():.4f}")
    print(f"   Loss requires grad: {loss.requires_grad}")

    # 6. Test Training Loop (1 epoch)
    print("\n🏋️ Testing Training Loop...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    total_loss = 0

    for batch_idx, (batch_sequences, batch_structures) in enumerate(train_loader):
        print(f"   Batch {batch_idx}: Processing {len(batch_sequences)} sequences")
        print(f"   Sequence lengths: {[len(seq) for seq in batch_sequences]}")

        optimizer.zero_grad()
        batch_loss = 0

        for seq, A_true in zip(batch_sequences, batch_structures):
            trajectory = model(seq)

            seq_loss = 0
            for t, A_t in enumerate(trajectory):
                seq_loss += differentiable_f1_loss(A_t, A_true)
            seq_loss /= len(trajectory)
            batch_loss += seq_loss

        batch_loss /= len(batch_sequences)
        batch_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        print(
            f"   Batch {batch_idx}: Loss={batch_loss.item():.4f}, Grad_norm={grad_norm:.4f}"
        )

    print(
        f"✅ Training step completed. Average loss: {total_loss / len(train_loader):.4f}"
    )

    # 7. Test Prediction Quality
    print("\n🎯 Testing Prediction Quality...")
    model.eval()
    with torch.no_grad():
        for i, (seq, true_structure) in enumerate(zip(sequences[:2], structures[:2])):
            trajectory = model(seq)
            pred_structure = trajectory[-1]

            pred_pairs = torch.sum(pred_structure > 0.5).item()
            true_pairs = torch.sum(dataset.structures[i]).item() // 2

            print(f"   Sample {i + 1}:")
            print(f"     Sequence: {seq} (len: {len(seq)})")
            print(f"     True structure: {true_structure}")
            print(f"     True pairs: {true_pairs}, Predicted pairs: {pred_pairs}")

    print("\n🎉 End-to-End Test Complete!")
    return model, train_loader


if __name__ == "__main__":
    model, train_loader = test_end_to_end()

    # Optional: Run a few training epochs
    print("\n🚀 Running 3 training epochs...")
    train_e2efold(model, train_loader, num_epochs=3, lr=0.001)
