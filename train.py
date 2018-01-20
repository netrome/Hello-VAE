import numpy as np
import torch
import torch.nn.functional as F

from torch.autograd import Variable
import torch.utils.data.dataloader as dataloader

from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms

import utils
import networks

dataset = FashionMNIST("~/BigData/fashion_mnist/", download=True, transform=transforms.ToTensor())
data_loader = dataloader.DataLoader(dataset, batch_size=141,)
visualizer = utils.Visualizer()

img = dataset[10]


# Initialize networks
encoder = networks.Encoder()
decoder = networks.Decoder()


def sample_with_reparametrization_trick(mu, log_var):
    std = log_var.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    return eps.mul(std).add_(mu)


for i, batch in enumerate(data_loader):

    if i % 100 == 0:
        patterns = Variable(batch[0])
        print(encoder(patterns))
        encoded = encoder(patterns)
        print()
        decoded = decoder(encoded[0])
        print(F.sigmoid(decoded))
        visualizer.batch_images(batch[0])
        break

