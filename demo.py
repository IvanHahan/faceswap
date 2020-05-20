import torch
import cv2
import argparse
from datasets import MyDataset, MyDatasetSampler
from utils.path import abs_path
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
from unet import UNet
import numpy as np
from utils import image_processing, landmarks as landmarks_processing

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default=abs_path('data/my/images/0.png'))
parser.add_argument('--landmarks_path', default=abs_path('data/my/landmarks/20.npy'))
parser.add_argument('--device', default='cpu')

args = parser.parse_args()

image = cv2.imread(args.image_path)
landmarks = np.load(args.landmarks_path)

plt.imshow(image)
plt.show()

heatmaps = landmarks_processing.landmarks2heatmaps(landmarks, image.shape[:2])
heatmaps = landmarks_processing.resize_heatmaps(heatmaps, 128)
heatmaps = landmarks_processing.pad_heatmaps(heatmaps, (128, 128))
hm = landmarks_processing.heatmaps2image(heatmaps)
heatmaps = np.expand_dims(heatmaps, 0)

image = image_processing.resize_image(image, 128)
image = image_processing.pad_image(image, (128, 128))[0]
image = image_processing.normalize_image(image)
image = image.transpose([2, 0, 1])
image = np.expand_dims(image, 0)

plt.imshow(hm)
plt.show()

image = torch.from_numpy(image).float()
heatmaps = torch.from_numpy(heatmaps).float()

generator = UNet(71, 3, False).to('cpu')
generator.load_state_dict(torch.load('model/generator19.pth', map_location='cpu'))
generator.train(False)

gen_in = torch.cat([image, heatmaps], 1)

gen_out = generator(gen_in)

out = gen_out.detach().cpu().numpy()[0].transpose([1, 2, 0]) * 127.5 + 127.5
out = cv2.cvtColor(out.astype('uint8'), cv2.COLOR_BGR2RGB)

plt.imshow(out)
plt.show()
