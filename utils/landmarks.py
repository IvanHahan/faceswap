import numpy as np
from utils.image_processing import resize_image, pad_image
import cv2


def landmarks2heatmaps(landmarks, shape):
    heatmaps = []
    for l in landmarks:
        canvas = np.zeros(shape[:2], dtype='float32')
        cv2.circle(canvas, (l[0], l[1]), 5, 1, -1)
        heatmaps.append(canvas)
    landmarks = np.array(heatmaps)
    return landmarks


def resize_heatmaps(heatmaps, size):
    resized_landmarks = []
    for l in heatmaps:
        l = resize_image(l, size)
        resized_landmarks.append(l)
    heatmaps = np.array(resized_landmarks)
    return heatmaps


def pad_heatmaps(heatmaps, size):
    padded_landmarks = []
    for l in heatmaps:
        l = pad_image(l, size)[0]
        padded_landmarks.append(l)
    heatmaps = np.array(padded_landmarks)
    return heatmaps


def heatmaps2image(heatmaps):
    heatmap = np.bitwise_or.reduce(heatmaps > 0, axis=0)
    return heatmap
