import torch
from torch import nn


class CriticNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super().__init__()

        self.fc = None

    def forward(self, state):
        x = self.fc(state)
        return x
