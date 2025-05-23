import torch
from rna_folding import RNA3DFolding


def main():
    sequence = "GGGGGCCACAGCAGAAGCGUUCACGUCGCAGCCCCUGUCAGCCAUUGCACUCCGGCUGCGAAUUCUGCU"
    embeddings_diemension = 64

    model = RNA3DFolding(d_model=embeddings_diemension)

    with torch.no_grad():
        score_matrix = model(sequence)

    print(score_matrix)


if __name__ == "__main__":
    main()
