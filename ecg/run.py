from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch

from model import Model

import numpy as np
import itertools
import pickle

EPOCH = 200
KERNEL_SIZE = 3
POOLING_SIZE = 2
BATCH_SIZE = 64

DATA_PATH = "./pickle/"
DEVICE = torch.device("mps")

def list_to_list(input_list):
    input_list_to_list = list(itertools.chain(*input_list))
    return input_list_to_list

record_list = []
pickle_input = dict()
X, y = [], []

print("[INFO] Read records file from ", DATA_PATH)
with open(DATA_PATH + 'RECORDS') as f:
    record_lines = f.readlines()

for i in range(len(record_lines)):
    record_list.append(str(record_lines[i].strip()))

for i in tqdm(range(len(record_list))):
    temp_path = DATA_PATH + "mit" + record_list[i] + ".pkl"
    with open(temp_path, 'rb') as f:
        pickle_input = pickle.load(f)
        for i in range(len(pickle_input[0])):
            X.append(pickle_input[0][i])

        for i in range(len(pickle_input[1])):
            check_ann = pickle_input[1][i]
            temp_ann_list = list()
            if check_ann == "N":            # Normal
                temp_ann_list.append(0)

            elif check_ann == "S":          # Supra-ventricular
                temp_ann_list.append(1)

            elif check_ann == "V":          # Ventricular
                temp_ann_list.append(2)

            elif check_ann == "F":          # False alarm
                temp_ann_list.append(3)

            else:                           # Unclassed 
                temp_ann_list.append(4)
            y.append(temp_ann_list)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33, random_state=42, shuffle=True)

class TrainDataset(Dataset):
    def __init__(self):
        self.X = X_train
        self.y = y_train
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor(self.y[idx])
        return X, y

class TestDataset(Dataset):
    def __init__(self):
        self.X = X_test
        self.y = y_test
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor(self.y[idx])
        return X, y

class ValidationDataset(Dataset):
    def __init__(self):
        self.X = X_val
        self.y = y_val
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.FloatTensor(self.X[idx])
        y = torch.FloatTensor(self.y[idx])
        return X, y
    
train_dataset = TrainDataset()
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = TestDataset()
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

validation_dataset = ValidationDataset()
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Model().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()

print(model)

def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (x_data, y_data) in enumerate(train_loader):
        x_data = x_data.to(DEVICE).view(64, -1)
        y_data = y_data.to(DEVICE)

        print(x_data.size())

        optimizer.zero_grad()
        output = model(x_data)

        print(y_data.size())
        print(output.size())

        loss = criterion(output, y_data)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{}({:.0f}%)]\tTrain Loss: {:.6F}".format(Epoch, batch_idx * len(x_data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
        
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for x, y in validation_dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            output = model(x)
            test_loss += criterion(output, y).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(y.view_as(prediction)).sum().item()
        
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

for Epoch in range(1, EPOCH + 1):
    train(model, train_dataloader, optimizer, log_interval=200)
    test_loss, test_accuracy = evaluate(model, test_dataloader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %\n".format(Epoch, test_loss, test_accuracy))