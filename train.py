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
from tqdm import tqdm
from matplotlib import pyplot as plt
import argparse
import cv2
from utils.path import make_dir_if_needed
from ranger import ranger



parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=abs_path('data/my'))
parser.add_argument('--device', default='cpu')
parser.add_argument('--verbosity', default=1, type=int)
parser.add_argument('--l_triple', default=1, type=float)
parser.add_argument('--l_adv', default=0.01, type=float)
args = parser.parse_args()

np.random.seed(12)
random.seed(12)
torch.manual_seed(12)
epochs = 100
steps_discriminator = 3
device = args.device
verbosity = args.verbosity

make_dir_if_needed('images')

generator = UNet(71, 3, False).to(device)
discriminator = Discriminator(3, 64).to(device)

gl_data_sampler = MyDatasetSampler(args.data_dir, device, size=64)
disc_data_sampler = MyDatasetSampler(args.data_dir, device, length=3, size=64)

gen_optim = ranger(generator.parameters())
disc_optim = ranger(discriminator.parameters())

losses = []

for e in range(epochs):
    print('EPOCH {}'.format(e))
    for first, second, third in tqdm(DataLoader(gl_data_sampler, batch_size=1)):

        discriminator.train(True)
        generator.train(True)

        for d_first, d_second, _ in DataLoader(disc_data_sampler, batch_size=1):

            disc_optim.zero_grad()
            gen_in = torch.cat([d_first[0], d_second[1]], 1)

            gen_out = generator(gen_in)
            true_out = discriminator(d_first[0]).view((-1))
            fake_out = discriminator(gen_out.detach()).view((-1))
            pred = torch.cat((fake_out, true_out), 0)
            label = torch.Tensor([0, 1]).to(device)
            bce_loss = F.binary_cross_entropy(pred, label)
            bce_loss.backward()
            disc_optim.step()

        gen_optim.zero_grad()

        gen_out = generator(torch.cat([first[0], second[1]], 1))
        fake_out = discriminator(gen_out).view((-1))

        label = torch.Tensor([1]).to(device)

        adv_loss = F.binary_cross_entropy(fake_out, label)

        I_ = gen_out.detach()
        GI = generator(torch.cat([first[0], third[1]], 1))
        GI_ = generator(torch.cat([I_, third[1]], 1))

        triple_loss = torch.square(GI_ - GI).mean()

        loss = args.l_triple * triple_loss + args.l_adv * adv_loss
        losses.append(loss.item())

        loss.backward()
        gen_optim.step()

    print(losses[-1])
    if e % verbosity == 0:
        fst = first[0].cpu().numpy()[0].transpose([1, 2, 0]) * 127.5 + 127.5
        fst = fst.astype('uint8')
        snd = second[0].cpu().numpy()[0].transpose([1, 2, 0]) * 127.5 + 127.5
        snd = snd.astype('uint8')
        out = gen_out.detach().cpu().numpy()[0].transpose([1, 2, 0]) * 127.5 + 127.5
        out = out.astype('uint8')
        cv2.imwrite(f'images/{e}.png', out)
        plt.imshow(fst)
        plt.show()
        plt.imshow(snd)
        plt.show()
        plt.imshow(out)
        plt.show()







