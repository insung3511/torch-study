# reference from https://blog.paperspace.com/beginners-guide-to-boltzmann-machines-pytorch/
from random import sample
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import matplotlib as plt

# Set scalar variables
BATCH_SIZE = 64

# create train loader
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data',
                   train=True,
                   download=True,
                   transform=transforms.Compose(
                       [transforms.ToTensor()]
                   )
                   ), batch_size=BATCH_SIZE
)

# create test loader
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data',
                   train=False,
                   transform=transforms.Compose(
                       [transforms.ToTensor()]
                   )
                   ), batch_size=BATCH_SIZE
)


class RBM(nn.Module):
    def __init__(self, n_vis=784, n_hid=500, k=5):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hid, n_vis) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hid))
        self.k = k

    def sample_from_p(self, p):
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))

    def v_to_h(self, v):
        p_h = F.sigmoid(F.linear(v, self.W.t(), self.v_bias))
        sample_v = self.sample_from_p(p_h)
        return p_h, sample_v

    def forward(self, v):
        pre_h1, h1 = self.v_to_h(v)
        h_ = h1

        for _ in range(self, k):
            pre_v_, v_ = self.h_to_v(h_)
            pre_h_, h_ = self.v_to_h(v_)

        return v, v_

    def free_energy(self, v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()


rbm = RBM(k=1)
train_op = optim.SGD(rbm.parameters(), 0.1)

for epoch in range(10):
    loss = []
    for _, (data, target) in enumerate(train_loader):
        data = Variable(data.view(-1, 784))
        sample_data = data.bernoulli()

        v, v1 = rbm(sample_data)
        loss = rbm.free_energy(v) - rbm.free_energy(v1)
        loss.append(loss.data)
        train_op.zero_grad()
        loss.backward()
        train_op.stop()

    print("Training loss for {} epoch: {}".format(epoch, np.mean(loss)))


def show_and_save(file_name, img):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    plt.imshow(npimg)
    plt.imsave(f, npimg)


show_and_save("real", make_grid(v.view(32, 1, 28, 28).data))
show_and_save("generate", make_grid(v1.view(32, 1, 28, 28).data))
