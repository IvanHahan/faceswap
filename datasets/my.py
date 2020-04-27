import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.image_processing import resize_image, pad_image
from utils.path import abs_path


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
        canvas = np.zeros(image.shape[:2], dtype='float32')
        for l in landmarks:
            canvas[l[1], l[0]] = 1
        landmarks = canvas

        image, landmarks = self._resize_sample(image, landmarks)
        image, landmarks = self._pad_sample(image, landmarks)
        image = (image - 127.5) / 127.5

        image = np.transpose(image, [2, 0, 1])
        landmarks = np.expand_dims(landmarks, 0)

        return image, landmarks

    def _resize_sample(self, image, landmarks):
        image = resize_image(image, self.size)
        landmarks = resize_image(landmarks, self.size)
        return image, landmarks

    def _pad_sample(self, image, landmarks):
        image = pad_image(image, (self.size, self.size))[0]
        landmarks = pad_image(landmarks, (self.size, self.size))[0]
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
        first = [torch.from_numpy(x).to(self.device) for x in first]
        second = [torch.from_numpy(x).to(self.device) for x in second]

        return first, second

    def __len__(self):
        if self.length is None:
            return super().__len__()
        return self.length


if __name__ == '__main__':
    dataset = MyDatasetSampler(abs_path('data/my'))
    loader = torch.utils.data.DataLoader(dataset)
    for i in loader:
        print(dataset[0])
