import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.image_processing import resize_image, pad_image
from utils.path import abs_path
import matplotlib.pyplot as plt
import numpy as np


class MyDataset(Dataset):

    def __init__(self, dir, size=64):
        self.size = size
        self.image_dir = os.path.join(dir, 'images')
        self.image_names = np.array(os.listdir(self.image_dir))
        self.landmarks_dir = os.path.join(dir, 'landmarks')
        self.landmarks_names = np.array(os.listdir(self.landmarks_dir))
        np.random.shuffle(self.image_names)
        np.random.shuffle(self.landmarks_names)

    def __getitem__(self, item):
        image = cv2.imread(os.path.join(self.image_dir, '{}.png'.format(item))).astype('float32')
        landmarks = np.load(os.path.join(self.landmarks_dir, '{}.npy'.format(item)))
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
        return len(os.listdir(self.image_dir))


class MyDatasetSampler(MyDataset):

    def __init__(self, dir, device, length=None, size=64):
        super().__init__(dir, size)
        self.length = length
        self.device = device

    def __getitem__(self, _):
        first = super().__getitem__(np.random.randint(0, len(self)))
        second = super().__getitem__(np.random.randint(0, len(self)))
        third = super().__getitem__(np.random.randint(0, len(self)))
        first = [torch.from_numpy(x).to(self.device) for x in first]
        second = [torch.from_numpy(x).to(self.device) for x in second]
        third = [torch.from_numpy(x).to(self.device) for x in third]

        return first, second, third

    def __len__(self):
        if self.length is None:
            return super().__len__()
        return self.length


if __name__ == '__main__':
    dataset = MyDatasetSampler(abs_path('data/my'), 'cpu')
    loader = torch.utils.data.DataLoader(dataset)
    for i in loader:
        fst = i[0][0].numpy()[0].transpose([1, 2, 0]) * 127.5
        fst += 127.5
        fst = fst.astype('uint8')
        mask = i[0][1].numpy()[0]
        mask = np.bitwise_or.reduce(mask > 0, axis=0).astype('uint8')
        plt.imshow(mask)
        plt.show()
        plt.imshow(fst)
        plt.show()
