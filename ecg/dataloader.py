from cgi import test
from logging import root
from torchvision.transforms import ToTensor, Lambda
from torchvision import datasets


class Dataset:
    def __init__(self):
        self.root = root

    def train_data(root):
        training_data = datasets.FashionMNIST(
            root=root, train=True, download=True, transform=ToTensor()
        )

        return training_data

    def test_data(root):
        test_data = datasets.FashionMNIST(
            root=root, train=False, download=True, transform=ToTensor()
        )

        return test_data
