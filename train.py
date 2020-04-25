import random

import numpy as np
import torch

from datasets import MyDataset
from discriminator import Discriminator
from unet import UNet
from utils.path import abs_path
from torch.utils.data import DataLoader
from torch import nn
from torch import functional as F

np.random.seed(12)
random.seed(12)
torch.manual_seed(12)
steps = 100
steps_discriminator = 3

generator = UNet(4, 3, False)
discriminator = Discriminator(3, 64)
pipeline = nn.Sequential(
    generator, discriminator
)

dataset = MyDataset(abs_path('data/my'))
data_loader = DataLoader(dataset, batch_size=1)

for i in range(steps):

    generator.train(False)
    for i in range(steps_discriminator):
        first = dataset[np.random.randint(0, len(dataset))]
        second = dataset[np.random.randint(0, len(dataset))]
        gen_in = np.concatenate(first[0], second[1], axis=0)

        gen_out = generator(gen_in)
        true_out = discriminator(first[0])
        fake_out = discriminator(gen_out)
        bce_loss = F.binary_cross_entropy(fake_out, 0) + F.binary_cross_entropy(true_out, 1)
        bce_loss.backward()

    first = dataset[np.random.randint(0, len(dataset))]
    second = dataset[np.random.randint(0, len(dataset))]
    gen_in = np.concatenate(first[0], second[1], axis=0)

    generator.train(True)

    gen_out = generator(gen_in)
    fake_out = discriminator(gen_out)

    gen_rec_out = generator(gen_out)
    fake_rec_out = discriminator(gen_rec_out)







