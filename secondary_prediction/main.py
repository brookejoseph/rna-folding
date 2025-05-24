import torch
import torch.nn as nn
from forward import RNA3DFolding
from backwards import PostProcessingNetwork
from train import train_e2efold


class E2Efold(nn.Module):
    def __init__(self, max_seq_len=512, d_model=64, max_iterations=20):
        super().__init__()
        self.deep_score_net = RNA3DFolding(max_seq_len, d_model)
        self.post_process_net = PostProcessingNetwork(max_iterations)

    def forward(self):
        sequence = "UUGAGAGAACUCGGGUGAAGGAACUAGGCAAAAUGGUGCCGUAACUUCGGGAGAAGGCACGCUGAUAUGUAGGUGAGGUCCCUCGCGGAUGGAGCUGAAAUCAGUCGAAGAUACCAGCUGGCUGCAACUGUUUAUUAAAAACACAGCACUGUGCAAACACGAAAGUGGACGUAUACGGUGUGACGCCUGCCCGGUGCCGGAAGGUUAAUUGAUGGGGUUAGCGCAAGCGAAGCUCUUGAUCGAAGCCCCGGUAAACGGCGGCCGUAACUAUAACGGUCCUAAGGUAGCGAAAUUCCUUGUCGGGUAAGUUCCGACCUGCACGAAUGGCGUAAUGAUGGCCAGGCUGUCUCCACCCGAGACUCAG"
        scores = self.deep_score_net(sequence)
        print("scores ", scores)

        trajectory = self.post_process_net(scores, sequence)

        return trajectory


model = E2Efold()
model.forward()
