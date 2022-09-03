import torch
import torch.nn as nn
from dataloader import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Lambda, ToTensor

BATCH_SIZE = 64
EPOCH = 100


train_dataset = datasets.FashionMNIST(
    root="../data/FashionMNIST",
    train=True,
    download=True,
    transform=transforms.ToTensor(),
)

test_dataset = datasets.FashionMNIST(
    root="../data/FashionMNIST", train=False, transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False
)


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(512, 256, (3, 3), (2, 2)),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.stack(x)
        return logits


model = Model()
print(model)
