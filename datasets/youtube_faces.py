import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.image_processing import resize_image, pad_image
from utils.path import abs_path
import matplotlib.pyplot as plt
import numpy as np


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

        image = cv2.imread(os.path.join(dir, 'frames/{}.png'.format(index))).astype('float32')
        landmarks = np.load(os.path.join(dir, 'landmarks/{}.npy'.format(index)))
        max_landmark_x = np.max(landmarks[:, 0]) + 1
        max_landmark_y = np.max(landmarks[:, 1]) + 1
        image = pad_image(image, (max_landmark_x, max_landmark_y))[0]
        heatmaps = []
        for l in landmarks:
            canvas = np.zeros(image.shape[:2], dtype='float32')
            cv2.circle(canvas, (l[0], l[1]), 5, 1, -1)
            heatmaps.append(canvas)
        landmarks = np.array(heatmaps)

        image, landmarks = self._resize_sample(image, landmarks)
        image, landmarks = self._pad_sample(image, landmarks)
        image = (image - 127.5) / 127.5

        image = np.transpose(image, [2, 0, 1])

        return image, landmarks

    def sample_triplet(self, dir):
        first = self.get_sample(np.random.randint(0, len(self)), dir)
        second = self.get_sample(np.random.randint(0, len(self)), dir)
        third = self.get_sample(np.random.randint(0, len(self)), dir)
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
