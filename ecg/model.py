import torch.nn.functional as F
import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1d_a = nn.Conv1d(64, 64, kernel_size=2, stride=2)
        self.softmax  = nn.Softmax(1)

    def forward(self, x):
        x = self.conv1d_a(x)
        x = self.softmax(x)

        return x