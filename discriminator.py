import torch
from torch import nn
import numpy as np

nc = 3
ndf = 64


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, in_shape=64, start_filters=64):
        super(Discriminator, self).__init__()
        self.in_block = nn.Sequential(
            nn.Conv2d(in_channels, start_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.ModuleList()
        while in_shape != 8:
            self.blocks.append(nn.Sequential(
                nn.Conv2d(start_filters, start_filters * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(start_filters * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            start_filters *= 2
            in_shape /= 2
        self.out_block = nn.Sequential(
            nn.Conv2d(start_filters, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.in_block(x)
        for b in self.blocks:
            x = b(x)
        return self.out_block(x)


if __name__ == '__main__':
    model = Discriminator()
    input = torch.from_numpy(np.ones((1, 3, 64, 64))).float()
    print(model)
    print(model(input).shape)
