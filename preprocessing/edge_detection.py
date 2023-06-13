import pathlib

import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import skimage
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter, disk
from skimage.util import img_as_ubyte


folder = pathlib.Path('processed_data')
files = folder.rglob('*.png')

def augment_image(file):
    img = cv2.imread(file)

    img8 = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    grey8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
    img_canny = canny(grey8, sigma=2, low_threshold=20, high_threshold=30)

    # save img_canny
    new_file = file.replace('processed_data', 'processed_data_canny')
    skimage.io.imsave(new_file, img_canny)
    print(f'augmented {new_file}')

for file in files:
    augment_image(str(file))
