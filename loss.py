import torch
import torch.nn as nn


class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, target):
        loss = self.ce(self.softmax(input), target)
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, target):
        logp = self.ce(self.softmax(input), target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
