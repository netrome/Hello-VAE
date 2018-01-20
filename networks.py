import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 4, 4, stride=2, padding=1)  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(4, 16, 4, stride=2, padding=1)  # 14x14 -> 7x7

        self.conv3 = nn.Conv2d(16, 16, 3)  # 7x7 -> 5x5
        self.conv4 = nn.Conv2d(16, 16, 3)  # 6x6 -> 3x3

        self.to_mean = nn.Linear(144, 20)
        self.to_logvar = nn.Linear(144, 20)

        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4]

    def forward(self, batch):
        for conv in self.convs:
            batch = F.selu(conv(batch))
        flat = batch.view(-1, 144)
        return self.to_mean(flat), self.to_logvar(flat)


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(20, 400)

        self.deconv1 = nn.ConvTranspose2d(16, 16, 3)  # 5x5 -> 7x7
        self.deconv2 = nn.ConvTranspose2d(16, 4, 4, stride=2, padding=1)  # 7x7 -> 14x14
        self.deconv3 = nn.ConvTranspose2d(4, 1, 4, stride=2, padding=1)  # 14x14 -> 28x28

        self.deconvs = [self.deconv1, self.deconv2, self.deconv3]

    def forward(self, batch):
        batch = F.selu(self.fc(batch))
        img = batch.view(-1, 16, 5, 5)

        for deconv in self.deconvs:
            img = F.selu(deconv(img))

        return img



