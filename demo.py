import torch
import cv2
import argparse
from utils.path import abs_path
from matplotlib import pyplot as plt
from generator import UNet
from generator.utils import prepare_input, postprocess
import numpy as np
from utils import image_processing, landmarks as landmarks_processing

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default=abs_path('data/my/images/0.png'))
parser.add_argument('--landmarks_path', default=abs_path('data/my/landmarks/20.npy'))
parser.add_argument('--device', default='cpu')

args = parser.parse_args()

image = cv2.imread(args.image_path)
landmarks = np.load(args.landmarks_path)
hm = landmarks_processing.landmarks2image(landmarks, image.shape)

plt.imshow(image)
plt.show()
plt.imshow(hm)
plt.show()

generator = UNet(71, 3, False).to('cpu')
generator.load_state_dict(torch.load('model/generator19.pth', map_location='cpu'))
generator.train(False)

gen_in = prepare_input(image, landmarks)

gen_out = generator(gen_in)

out = postprocess(gen_out)

plt.imshow(out)
plt.show()
