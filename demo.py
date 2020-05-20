import torch
import cv2
import argparse
from utils.path import abs_path
from matplotlib import pyplot as plt
from generator import UNet
from generator.utils import prepare_input
import numpy as np
from utils import image_processing, landmarks as landmarks_processing

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default=abs_path('data/my/images/0.png'))
parser.add_argument('--landmarks_path', default=abs_path('data/my/landmarks/20.npy'))
parser.add_argument('--device', default='cpu')
parser.add_argument('--from_dict', action='store_false')

args = parser.parse_args()

image = cv2.imread(args.image_path)
landmarks = np.load(args.landmarks_path)
hm = landmarks_processing.landmarks2image(landmarks, image.shape)

plt.imshow(image)
plt.show()
plt.imshow(hm)
plt.show()


if args.from_dict:
    generator = UNet(71, 3, False).to('cpu')
    generator.load_state_dict(torch.load('model/generator19.pth', map_location='cpu'))
else:
    generator = torch.load('model/generator40.pth', map_location='cpu')
generator.train(False)

gen_in = prepare_input(image, landmarks)

gen_out = generator(gen_in)

out = gen_out.detach().cpu().numpy()[0].transpose([1, 2, 0]) * 127.5 + 127.5
out = cv2.cvtColor(out.astype('uint8'), cv2.COLOR_BGR2RGB)

plt.imshow(out)
plt.show()
