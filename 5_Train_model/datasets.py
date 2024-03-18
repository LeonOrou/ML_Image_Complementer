# -*- coding: utf-8 -*-
"""example_project/datasets.py

Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 15.04.2022

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

###############################################################################

Datasets file of example project.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import PIL
import random
# from line_profiler_pycharm import profile
import os
import os.path
import dill
from datareader import datareader
from pathlib import Path
from tqdm import tqdm
import random

'''
Notes:
After importing TransformsDataset to another file, use:

TransformsDataset(dataset=dataset)

for relatively small but many random transformations given in transform_chain, 
transform_chain is the default transformator for this class
'''


def add_normal_noise(input_tensor, mean: int = 0, std: float = 0.2):
    """Simple function that adds noise from a normal distribution to `input_tensor`"""
    # Create the tensor containing the noise
    noise_tensor = torch.empty_like(input_tensor)
    noise_tensor.normal_(mean=mean, std=std)
    # Add noise to input tensor and return results
    return input_tensor + noise_tensor


def wrap_add_normal_noise(mean: int = 0, std: float = 0.2):
    """Return function that calls add_normal_noise() and clamps values to range [0, 1]"""

    def noisy_image(input_tensor):
        input_tensor = add_normal_noise(input_tensor, mean, std)
        return torch.clamp(input_tensor, min=0, max=1)

    return noisy_image


# @profile
def get_transformator():
    # Create chain of transforms
    transform_chain = transforms.Compose([
        # transforms.Resize(size=120),  # resizing down for computation time
        # adding slight but many random (non-repeating) data augmentations
        transforms.ColorJitter(
            brightness=random.randint(0, 10)/100,
            saturation=random.randint(0, 10)/100),
        transforms.RandomRotation(degrees=random.randint(0, 20)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.Resize(size=100),
        transforms.CenterCrop(size=(100, 100)),  # resizing to same as eval set
        transforms.ToTensor()
    ])
    return transform_chain


def transform_image(image, transformator=get_transformator()):
    return transformator(image)


class TestSampler:
    offsets: np.ndarray
    spacings: np.ndarray

    input_array: np.ndarray
    known_array: np.ndarray

    sample_id: np.ndarray

    original_array: np.ndarray


class TrainSampler:
    offsets: np.ndarray
    spacings: np.ndarray

    input_array: np.ndarray
    known_array: np.ndarray

    target_array: np.ndarray

    original_array: np.ndarray


class TrainingDataset(Dataset):
    samples: TrainSampler()

    def __init__(self, transformator=get_transformator(), new_trainingset=False):
        self.transforms = transformator
        self.new_trainingset = new_trainingset
        self.__load_samples(new_trainingset=False)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        return sample.input_array, sample.known_array, sample.offsets, sample.spacings, index, sample.original_array

    def __len__(self):
        return len(self.samples)

    def __load_samples(self, new_trainingset: bool):
        # set new_trainingset=True if new trainingset should be generated
        # e.g. for transformation after each epoch
        self.samples: TrainSampler() = []

        train_set_path = os.path.abspath('training\\')
        files = []

        for path in Path(train_set_path).rglob('*.*'):
            files.append(str(path))

        if new_trainingset is True:
            files = files[:len(files)*3/5]

        dataset = np.empty(shape=(len(files), 3, 100, 100), dtype=float)
        transform_chain_raw = transforms.Compose(  # for test and validation set, default is training transformation
            [transforms.Resize(size=100),
             transforms.CenterCrop(size=(100, 100)),  # resizing to same as eval set
             transforms.ToTensor()])

        indices = int(len(files) / 5)
        train_indices = indices * 3  # first 17646 images are for training
        validation_indices = indices * 4  # indices between 17646 and 23528 are for validation, afterwards for test

        #
        # full dataset
        #
        # for i, image_path in tqdm(enumerate(files), desc="Processing files", total=len(files)):
        #     with PIL.Image.open(image_path) as img:
        #         if i < train_indices:  # apply transformations on testset, but not on val and test set
        #             transformed_image = transform_image(img, transform_chain_raw)
        #             dataset[i] = transformed_image
        #         elif i >= validation_indices:
        #             transformed_image = transform_image(img, transform_chain_raw)
        #             dataset[i] = transformed_image

        #
        # subset of dataset (for debugging or faster runtime): random n samples from all files
        #
        files = random.sample(files, 2000)

        for i, image_path in tqdm(enumerate(files), desc="Processing files", total=len(files)):
            with PIL.Image.open(image_path) as img:
                if i < train_indices:  # apply transformations on testset, but not on val and test set
                    transformed_image = transform_image(img, transform_chain_raw)
                    dataset[i] = transformed_image*255
                elif i >= validation_indices:
                    transformed_image = transform_image(img, transform_chain_raw)
                    dataset[i] = transformed_image*255

        for i in range(len(files)):
            sample = TrainSampler()
            sample.offsets = (random.randint(0, 8), random.randint(0, 8))
            sample.spacings = (random.randint(2, 6), random.randint(2, 6))
            array = np.transpose(dataset[i], (1, 2, 0))
            sample.input_array, sample.known_array, sample.target_array = datareader(array, sample.offsets, sample.spacings)
            sample.original_array = dataset[i]
            self.samples.append(sample)


class TestControlDataset(Dataset):

    samples: TestSampler()

    def __init__(self):
        self.__load_test_set()

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return sample.input_array, sample.known_array, sample.offsets, sample.spacings, sample.sample_id

    def __len__(self):
        return len(self.samples)

    def __load_test_set(self):
        self.samples = []

        with open(os.path.join('.', 'test', 'inputs.pkl'), 'rb') as f:
            test_set = dill.load(f)
            for i in range(len(test_set['input_arrays'])):
                sample = TestSampler()

                sample.offsets = test_set['offsets'][i]
                sample.spacings = test_set['spacings'][i]

                sample.input_array = test_set['input_arrays'][i]
                sample.known_array = test_set['known_arrays'][i]

                sample.sample_id = test_set['sample_ids'][i]

                self.samples.append(sample)
