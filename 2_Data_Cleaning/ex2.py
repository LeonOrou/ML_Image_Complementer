"""
Author: Leon Orou
Matr.Nr.: k12125027
Exercise 2
"""


import os

import PIL
from PIL import Image, ImageStat
import numpy as np
import glob
import hashlib
from tqdm import tqdm
from pathlib import Path
import re
import shutil


def validate_images(input_dir, output_dir, log_file, formatter):
    files = sorted(glob.glob(os.path.join(input_dir, "**", "*.jpg"), recursive=True))
    hashes = []  # used for all hash to compare values if already exists afterwards
    if not os.path.exists(os.path.abspath(output_dir)):
        os.mkdir(os.path.abspath(output_dir))
    if not os.path.exists(os.path.abspath(log_file)):
        os.mkdir(os.path.abspath(log_file))

    for i, image_file in tqdm(enumerate(files), desc="Processing files", total=len(files)):
        file_ending = re.findall("[.]\w+", files[i])[0]
        file_code = "{number:06}".format(number=i) + file_ending
        if files[i].lower().endswith(('.jpg', '.jpeg')):
            file_size = os.path.getsize(os.path.abspath(files[i]))
            if file_size <= 250000:
                # with np.array(Image.open(image_file)) as img:
                with PIL.Image.open(image_file) as img:
                    try:
                        img.verify()
                        width, height = img.size
                        channels = img.mode
                        if width*height >= 96 and channels == "RGB":
                            image = np.array(Image.open(image_file))
                            var_channels = image.var(axis=(0, 1))
                            if np.prod(var_channels) > 0:  # if no element is 0 (mul of elems)
                                hash_function = hashlib.sha256()
                                # some_data = bytes(img, encoding="utf-8")
                                # hash_function.update(some_data)
                                hash_value = hash_function.digest()
                                if hash_value not in hashes:
                                    hashes.append(hash_value)
                                    shutil.copy(image_file, output_dir)
                                    # output_file = open("/output_dir/output_file.txt", "w+")
                                    # output_file.write(file_code + "\n")
                                else:
                                    error_code = "The image was already added"
                                    write_log(file_code, error_code, input_dir)
                            else:
                                error_code = "At least one channel of the image always the same value"
                                write_log(file_code, error_code, input_dir)
                        else:
                            error_code = "image is smaller than 96 pixels or doesn't have the right RGB channels"
                            write_log(file_code, error_code, input_dir)
                    except Exception:
                        error_code = "Image could not be read"
                        write_log(file_code, error_code, input_dir)
            else:
                error_code = "image is larger than 250 kbit"
                write_log(file_code, error_code, input_dir)
        else:
            error_code = "image doesn't end with '.jpg', '.jpeg', '.JPG' or '.JPEG'"
            write_log(file_code, error_code, input_dir)


def write_log(file_code, error_code, input_dir):
    log_file = open(f"{input_dir}/log_file.txt", "a")
    log_file.write(file_code + ";" + error_code + "\n")


# path = r'.\unittest\unittest_input_0'
# output = validate_images(path, 'output_dir', 'log_dir')
# print(output)


