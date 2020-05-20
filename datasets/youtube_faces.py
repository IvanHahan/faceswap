import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.image_processing import resize_image, pad_image
from utils.path import abs_path
import matplotlib.pyplot as plt
import os


class YoutubeFaces(Dataset):

    def __init__(self, dir, size=64, len=None, device='cpu'):
        self.size = size
        self.names = os.listdir(dir)
        self.dir = dir
        self.device = device
        self.len = None
        if len is not None:
            self.len = len
        np.random.shuffle(self.names)

    def __getitem__(self, item):

        person_dir = os.path.join(self.dir, self.names[item])

        return self.sample_triplet(person_dir)

    def get_sample(self, index, dir):

        frame_name = os.listdir(os.path.join(dir, 'frames'))[index]

        image = cv2.imread(os.path.join(dir, 'frames/{}'.format(frame_name))).astype('float32')
        landmarks = np.load(os.path.join(dir, 'landmarks/{}.npy'.format(frame_name)))
        return image, landmarks

    def sample_triplet(self, dir):
        first = self.get_sample(np.random.randint(0, len(os.listdir(dir))), dir)
        second = self.get_sample(np.random.randint(0, len(os.listdir(dir))), dir)
        third = self.get_sample(np.random.randint(0, len(os.listdir(dir))), dir)
        first = [torch.from_numpy(x).to(self.device) for x in first]
        second = [torch.from_numpy(x).to(self.device) for x in second]
        third = [torch.from_numpy(x).to(self.device) for x in third]
        return first, second, third

    def _resize_sample(self, image, landmarks):
        image = resize_image(image, self.size)
        resized_landmarks = []
        for l in landmarks:
            l = resize_image(l, self.size)
            resized_landmarks.append(l)
        landmarks = np.array(resized_landmarks)
        return image, landmarks

    def _pad_sample(self, image, landmarks):
        image = pad_image(image, (self.size, self.size))[0]
        padded_landmarks = []
        for l in landmarks:
            l = pad_image(l, (self.size, self.size))[0]
            padded_landmarks.append(l)
        landmarks = np.array(padded_landmarks)
        return image, landmarks

    def __len__(self):
        if self.len:
            return self.len
        return len(self.names)
