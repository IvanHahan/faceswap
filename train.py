import random

import numpy as np
import torch

from datasets import MyDataset, MyDatasetSampler
from discriminator import Discriminator
from unet import UNet
from utils.path import abs_path
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F

np.random.seed(12)
random.seed(12)
torch.manual_seed(12)
steps = 100
steps_discriminator = 3
device = 'cpu'

generator = UNet(4, 3, False).to(device)
discriminator = Discriminator(3, 64).to(device)

gl_data_sampler = MyDatasetSampler(abs_path('data/my'), device, size=64)
disc_data_sampler = MyDatasetSampler(abs_path('data/my'), device, length=3, size=64)

gen_optim = torch.optim.Adam(generator.parameters())
disc_optim = torch.optim.Adam(discriminator.parameters())

for first, second in DataLoader(gl_data_sampler, batch_size=1):

    for d_first, d_second in DataLoader(disc_data_sampler, batch_size=1):

        disc_optim.zero_grad()
        gen_in = torch.cat([d_first[0], d_second[1]], 1)

        gen_out = generator(gen_in)
        true_out = discriminator(d_first[0]).view((-1))
        fake_out = discriminator(gen_out.detach()).view((-1))
        pred = torch.cat((fake_out, true_out), 0)
        label = torch.Tensor([0, 1])
        bce_loss = F.binary_cross_entropy(pred, label)
        bce_loss.backward()
        disc_optim.step()

    gen_optim.zero_grad()
    gen_in = torch.cat([first[0], second[1]], 1)

    gen_out = generator(gen_in)
    fake_out = discriminator(gen_out).view((-1))

    label = torch.Tensor([1])

    bce_loss = F.binary_cross_entropy(fake_out, label)
    bce_loss.backward()
    gen_optim.step()








