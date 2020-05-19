import random

import numpy as np
import torch

from datasets import YoutubeFaces
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
from percept_loss import PerceptualLoss
import os


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=os.environ['SM_CHANNEL_TRAIN'])
parser.add_argument('--vgg19_weights', default=os.environ['SM_CHANNEL_VGG19'])
parser.add_argument('--model_dir', default=os.environ['SM_MODEL_DIR'])

parser.add_argument('--device', default='cpu')
parser.add_argument('--verbosity', default=1, type=int)
parser.add_argument('--size', default=128, type=int)
parser.add_argument('--l_triple', default=100, type=float)
parser.add_argument('--l_adv', default=1, type=float)
parser.add_argument('--l_rec', default=100, type=float)
parser.add_argument('--l_pix', default=10, type=float)
parser.add_argument('--l_percept', default=10, type=float)
parser.add_argument('--epochs', default=20, type=float)
args = parser.parse_args()

np.random.seed(12)
random.seed(12)
torch.manual_seed(12)
epochs = args.epochs
steps_discriminator = 3
device = args.device
verbosity = args.verbosity

if __name__ == '__main__':

    make_dir_if_needed('images')
    make_dir_if_needed('model')

    generator = UNet(71, 3, False).to(device)
    discriminator = Discriminator(3, args.size).to(device)

    gl_data_sampler = YoutubeFaces(args.data_dir, device=device, size=args.size)
    disc_data_sampler = YoutubeFaces(args.data_dir, device=device, len=3, size=args.size)

    compute_perceptual = PerceptualLoss(args.vgg19_weights).to(device)

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

            pix_loss = torch.square(gen_out - second[0]).mean()

            percept_loss = compute_perceptual(gen_out, second[0])

            fake_out = discriminator(gen_out).view((-1))

            label = torch.Tensor([1]).to(device)

            adv_loss = F.binary_cross_entropy(fake_out, label)

            I_ = gen_out.detach()

            rec_I = generator(torch.cat([I_, first[1]], 1))

            rec_loss = torch.square(rec_I - first[0]).mean()

            GI = generator(torch.cat([first[0], third[1]], 1))
            GI_ = generator(torch.cat([I_, third[1]], 1))

            triple_loss = torch.square(GI_ - GI).mean()

            loss = args.l_triple * triple_loss + args.l_adv * adv_loss + \
                   args.l_rec * rec_loss + args.l_pix * pix_loss + args.l_percept * percept_loss
            losses.append(loss.item())

            loss.backward()
            gen_optim.step()

        print(losses[-1])
        if e % verbosity == 0:
            torch.save(generator, f'{args.model_dir}/generator{e}.pth')








