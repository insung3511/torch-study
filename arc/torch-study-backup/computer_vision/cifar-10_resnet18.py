import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms

DEVICE = torch.device("cpu")

print("Using PyTorch Version: ", torch.__version__, " Device: ", DEVICE)

BATCH_SIZE = 32
EPOCHS = 10

model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model = model.cpu()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

data_transform = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    ),
}

image_datasets = {
    x: datasets.ImageFolder("../data/hymenoptera_data", data_transform[x])
    for x in ["train", "val"]
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=BATCH_SIZE, num_workers=0, shuffle=True
    )
    for x in ["train", "val"]
}

for (X_train, y_train) in dataloaders["train"]:
    print("X_train: ", X_train.size(), "type: ", X_train.type())
    print("y_train: ", y_train.size(), "type: ", y_train.type())
    break


def train(model, train_loader, optimizer, log_interval):
    model.train()
    for (
        batch_idx,
        (image, label),
    ) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: ? [{batch_idx * len(image)} / {len(train_loader.dataset)}, {100. * batch_idx / len(train_loader):.5f}% ] Loss: {loss.item():.5f}"
            )


def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100.0 * correct / len(test_loader.dataset)

        print(f"Test Epoch: ?\tAccuracy: {test_accuracy:.5f}\tLoss: {test_loss:.5f}")
        return test_loss, test_accuracy
