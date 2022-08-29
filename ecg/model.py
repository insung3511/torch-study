from torchvision.transforms import ToTensor, Lambda
from torchvision import datasets

from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from dataloader import Dataset

BATCH_SIZE = 64
EPOCH = 100

train_dataloader = DataLoader(Dataset.train_data("../data/"), batch_size=BATCH_SIZE)
test_dataloader  = DataLoader(Dataset.test_data("../data/"), batch_size=BATCH_SIZE)

class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.stack   = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),

            nn.Conv2d(512, 512, (3, 3), (2, 2)),
            nn.ReLU(),

            nn.Conv2d(512, 256, (3, 3), (2, 2)),
            nn.ReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits
    
model = Model()
print(model)