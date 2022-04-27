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
import logging
from os import listdir


def validate_images(input_dir, output_dir, log_file, formatter):
    file_count = 0
    files = []
    #files = sorted(glob.glob(os.path.join(os.path.abspath(input_dir), "**", "*"), recursive=True))
    for path in Path(input_dir).rglob('*.*'):
        files.append(str(path))
    files.sort()
    hashes = []  # used for all hash to compare values if already exists afterwards
    if os.path.exists(os.path.abspath(output_dir)):
        output_path = os.path.abspath(output_dir)
        shutil.rmtree(output_path)
        os.makedirs(output_path)
    else:
        output_path = os.path.abspath(output_dir)
        os.makedirs(output_path)
    open(f"{os.path.abspath(log_file)}", 'w').close()
    logging.basicConfig(filename=log_file, filemode='w')

    for i, image_file in tqdm(enumerate(files), desc="Processing files", total=len(files)):
        file_ending = re.search("[.]\w+", files[i]).group()
        rel_file_path = image_file.replace(input_dir+"\\", "")
        if formatter == "06d":
            file_code = "{number:06}".format(number=file_count) + file_ending
        else:
            file_code = str(file_count) + str(file_ending)
        if files[i].lower().endswith(('.jpg', '.jpeg')):
            file_size = os.path.getsize(os.path.abspath(files[i]))
            if file_size <= 250000:
                try:
                    with PIL.Image.open(image_file) as img:
                        try:
                            img.verify()
                            width, height = img.size
                            channels = img.mode
                            if width >= 96 and height >= 96 and channels == "RGB":
                                image = np.array(Image.open(image_file))
                                var_channels = image.var(axis=(0, 1))
                                if np.prod(var_channels) > 0:  # if no element is 0 (mul of elems)
                                    hash_function = hashlib.sha256()
                                    hash_function.update(image)
                                    # some_data = bytes(img, encoding="utf-8")
                                    # hash_function.update(some_data)
                                    hash_value = hash_function.digest()
                                    if hash_value not in hashes:
                                        hashes.append(hash_value)
                                        #shutil.copy(file_code, output_path)
                                        #file_code = "{number:06}".format(number=file_count) + file_ending
                                        shutil.copy(image_file, output_path)
                                        pre, ext = os.path.splitext(file_code)
                                        file_name = os.path.basename(os.path.normpath(image_file))
                                        #file_name = re.search(r"[\\](.+)", image_file).group(-1)
                                        old_name = os.path.abspath(output_path) + "\\" + file_name
                                        new_name = str(os.path.abspath(output_path) + "\\" + pre + ".jpg")
                                        os.rename(old_name, new_name)
                                        file_count += 1
                                    else:
                                        error_code = "6"
                                        write_log(rel_file_path, error_code, log_file)
                                        continue
                                else:
                                    error_code = "5"
                                    write_log(rel_file_path, error_code, log_file)
                                    continue
                            else:
                                error_code = "4"
                                write_log(rel_file_path, error_code, log_file)
                                continue
                        except Exception as e:
                            error_code = f"3"
                            write_log(rel_file_path, error_code, log_file)
                            continue
                except Exception as e:
                    error_code = f"3"
                    write_log(rel_file_path, error_code, log_file)
                    continue
            else:
                error_code = "2"
                write_log(rel_file_path, error_code, log_file)
                continue
        else:
            error_code = "1"
            write_log(rel_file_path, error_code, log_file)
            continue
    return file_count


def write_log(rel_file_path, error_code, log_file):
    if log_file.endswith(".log"):
        log_file = open(f"{os.path.abspath(log_file)}", "a")
        log_file.write(rel_file_path + ";" + error_code + "\n")
    else:
        log_file = open(f"{os.path.abspath(log_file)}", "a")
        log_file.write(rel_file_path + ";" + error_code + "\n")


# input_dir = 'unittest\\unittest_input_1'
# log_file = 'unittest\\outputs\\unittest_input_1.log'
# output_dir = 'unittest\\outputs\\unittest_input_1'
# output = validate_images(input_dir, output_dir, log_file, formatter="06d")
# print(output)
#
#
