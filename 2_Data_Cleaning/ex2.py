"""
Author: Leon Orou
Matr.Nr.: k12125027
Exercise 2
"""


import os
from pathlib import Path


def validate_images(input_dir, output_dir, log_file):
    if not os.path.exists(os.path.abspath(output_dir)):
        os.mkdir(os.path.abspath(output_dir))
    if not os.path.exists(os.path.abspath(log_file)):
        os.mkdir(os.path.abspath(log_file))

    files = get_files(input_dir)

    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg')):
            if os.path.getsize(os.path.abspath(file)) >= 250000:
                # TODO check with pillow module if image is valid file without raising exception
                pass


def get_files(input_dir):
    files = []
    contents = os.listdir(input_dir) # read the contents of dir
    for item in contents:      # loop over those contents
        if os.path.isdir(item):
            get_files(item)     # recurse on subdirectories
        else:
            files.append(item)
    return files


path = r'.\unittest\unittest\unittest_input_0'
validate_images(path, 'output_dir', 'log_dir')



