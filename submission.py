
import numpy as np
import cv2
from data import image_cols, image_rows


def prep(img):
    img = img.astype('float32')
    img = cv2.threshold(img, 0.50, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    img = cv2.resize(img, (image_cols, image_rows))
    return img


def run_length_enc(label):