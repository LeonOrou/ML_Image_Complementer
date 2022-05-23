"""
Author: Leon Orou
Matr.Nr.: k12125027
Exercise 3
"""

import os
import torch
from PIL import Image, ImageStat
import numpy as np
import glob
import path
from path import Path
from pathlib import Path
from tqdm import tqdm


class ImageStandardizer:

    def __init__(self, input_dir: str):
        self.mean = None
        self.std = None
        files = []
        for path in Path(input_dir).rglob('*.jpg'):
            files.append(str(path))
        # self.files = sorted(glob.glob(os.path.join(os.path.abspath(input_dir), "*.jpg"), recursive=True))
        self.files = files
        if len(self.files) == 0:
            raise ValueError("No .jpg files in inout_dir")

    def analyze_images(self):

        means = np.zeros(shape=(len(self.files), 3), dtype=np.float64)
        stds = np.zeros(shape=(len(self.files), 3), dtype=np.float64)
        for i, image_file in tqdm(enumerate(self.files), desc="Processing files", total=len(self.files)):
            image = np.array(Image.open(image_file))
            means[i] = image.mean(axis=(0, 1))
            stds[i] = image.std(axis=(0, 1))
        self.mean = np.average(means, axis=0)
        self.std = np.average(stds, axis=0)

        # all_pixels = np.zeros((1410060, 3), dtype=np.float64)
        # pixel_counter = 0
        # for i, file in enumerate(self.files):
        #     image = np.array(Image.open(file))
        #     height, width, _ = image.shape
        #     for h_pixel in range(height):
        #         for w_pixel in range(width):
        #             all_pixels[pixel_counter] = image[h_pixel][w_pixel]
        #             pixel_counter += 1
        # self.mean = np.average(all_pixels, axis=0)
        # self.std = np.std(all_pixels, axis=0)

        # r_channel = []
        # g_channel = []
        # b_channel = []
        #
        # for i, file in enumerate(self.files):
        #     image = np.array(Image.open(file))
        #     height, width, _ = image.shape
        #     for h_pixel in range(height):
        #         for w_pixel in range(width):
        #             r_channel.append(image[h_pixel][w_pixel][0])
        #             g_channel.append(image[h_pixel][w_pixel][1])
        #             b_channel.append(image[h_pixel][w_pixel][2])
        # r_channel_mean = np.mean(r_channel)
        # g_channel_mean = np.mean(g_channel)
        # b_channel_mean = np.mean(b_channel)
        # self.mean = np.array([r_channel_mean, g_channel_mean, b_channel_mean], dtype=np.float64)
        #
        # r_channel_std = np.std(r_channel)
        # g_channel_std = np.std(g_channel)
        # b_channel_std = np.std(b_channel)
        # self.std = np.array([r_channel_std, g_channel_std, b_channel_std], dtype=np.float64)
        return self.mean, self.std

    def get_standardized_images(self):
        if self.mean is None or self.std is None:
            raise ValueError
        for file in self.files:
            image = np.array(Image.open(file), dtype=np.float32)
            assert len(image.shape) == 3 and image.shape[
                2] >= 3, f"{file}: {image.shape}"
            if image.shape[2] == 4:
                image = image[:, :, :-1]

            for pixel in range(0, len(image)):
                image[pixel] = np.subtract(image[pixel], self.mean, dtype=np.float64)
                image[pixel] = np.divide(image[pixel], self.std, dtype=np.float64)

            yield image
