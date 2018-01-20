import numpy as np
import torch
import torch.nn.functional as F

from torch.autograd import Variable
import torch.utils.data.dataloader as dataloader

from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms

import utils
import networks

BATCH_SIZE = 141

dataset = FashionMNIST("~/BigData/fashion_mnist/", download=True, transform=transforms.ToTensor())
data_loader = dataloader.DataLoader(dataset, batch_size=BATCH_SIZE,)
visualizer = utils.Visualizer()

img = dataset[10]


# Initialize networks
encoder = networks.Encoder()
decoder = networks.Decoder()

# Optimizer
optimizer = torch.optim.Adam([
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
])


def sample_with_reparametrization_trick(mu, log_var):
    std = log_var.mul(0.5).exp_()
    eps = Variable(std.data.new(std.size()).normal_())
    return eps.mul(std).add_(mu)


def loss_VAE(decoded, original, mu, log_var):
    MSE = torch.mean((decoded - original)**2)

    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    KLD /= BATCH_SIZE * 784
    return MSE + KLD


for epoch in range(10):
    for i, batch in enumerate(data_loader):

        patterns = Variable(batch[0])
        encoded = encoder(patterns)
        sampled = sample_with_reparametrization_trick(encoded[0], encoded[1])
        decoded = F.sigmoid(decoder(sampled))

        loss = loss_VAE(decoded, patterns, encoded[0], encoded[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Loss at {}: {}".format(i, loss))

    visualizer.batch_images(patterns.data, "real")
    visualizer.batch_images(decoded.data, "fake")


