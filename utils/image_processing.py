import cv2
import numpy as np


def resize_image(image, size=256):
    width = int(size * image.shape[1] / image.shape[0] if image.shape[0] > image.shape[1] else size)
    height = int(size * image.shape[0] / image.shape[1] if image.shape[0] < image.shape[1] else size)
    return cv2.resize(image, (width, height))


def pad_image(image, size=(256, 256)):
    dx = size[0] - image.shape[1]
    dy = size[1] - image.shape[0]

    dx = dx if dx >= 0 else 0
    dy = dy if dy >= 0 else 0

    value = ((0, dy), (0, dx), (0, 0)) if image.ndim == 3 else ((0, dy), (0, dx))

    return np.lib.pad(image, value, 'constant', constant_values=0), (dx, dy)


def normalize_image(image):
    return (image - 127.5) / 127.5
