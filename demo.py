import torch
import cv2
import argparse
from datasets import MyDataset, MyDatasetSampler
from utils.path import abs_path
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default=abs_path('data/my'))
parser.add_argument('--device', default='cpu')

args = parser.parse_args()

generator = torch.load('model/generator10.pth', map_location='cpu')
generator.train(False)

gl_data_sampler = MyDatasetSampler(args.data_dir, args.device, size=64)

for first, second, third in tqdm(DataLoader(gl_data_sampler, batch_size=1)):
    gen_in = torch.cat([first[0], second[1]], 1)

    gen_out = generator(gen_in)

    fst = first[0].cpu().numpy()[0].transpose([1, 2, 0]) * 127.5 + 127.5
    fst = fst.astype('uint8')
    snd = second[0].cpu().numpy()[0].transpose([1, 2, 0]) * 127.5 + 127.5
    snd = snd.astype('uint8')
    out = gen_out.detach().cpu().numpy()[0].transpose([1, 2, 0]) * 127.5 + 127.5
    out = out.astype('uint8')
    plt.imshow(fst)
    plt.show()
    plt.imshow(snd)
    plt.show()
    plt.imshow(out)
    plt.show()
