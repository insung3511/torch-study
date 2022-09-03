import torch.nn.functional as F
import torch.nn as nn
import torch

from dataloader import Dataset
from model import Model

model = Model()
datasets = Dataset()

train_data = datasets.train_data()
test_data = datasets.test_data()

