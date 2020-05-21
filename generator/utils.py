import torch
import cv2
import argparse
from utils.path import abs_path
from matplotlib import pyplot as plt
from generator import UNet
import numpy as np
from utils import image_processing, landmarks as landmarks_processing


def process_landmarks(landmarks, image_shape, target_size):
    max_x = max(image_shape[1], np.max(landmarks[:, 0]) + 1)
    max_y = max(image_shape[0], np.max(landmarks[:, 1]) + 1)
    heatmaps = landmarks_processing.landmarks2heatmaps(landmarks, (max_y, max_x))
    heatmaps = landmarks_processing.resize_heatmaps(heatmaps, target_size[0])
    heatmaps = landmarks_processing.pad_heatmaps(heatmaps, target_size)
    return heatmaps


def process_image(image, target_size):
    image = image_processing.resize_image(image, target_size[0])
    image = image_processing.pad_image(image, target_size)[0]
    image = image_processing.normalize_image(image)
    image = image.transpose([2, 0, 1])
    return image


def prepare_input(image, landmarks):
    image = process_image(image, (128, 128))
    heatmaps = process_landmarks(landmarks, image.shape, (128, 128))
    image = np.expand_dims(image, 0)
    heatmaps = np.expand_dims(heatmaps, 0)

    image = torch.from_numpy(image).float()
    heatmaps = torch.from_numpy(heatmaps).float()

    return torch.cat([image, heatmaps], 1)


def postprocess(output):
    out = output.detach().cpu().numpy()[0].transpose([1, 2, 0]) * 127.5 + 127.5
    out = cv2.cvtColor(out.astype('uint8'), cv2.COLOR_BGR2RGB)
    return out