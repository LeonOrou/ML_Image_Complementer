"""
Author: Leon Orou
Matr.Nr.: k12125027
Exercise 4
"""

import numpy as np
import math


def ex4(img_arr_edit, offset, spacing):
    if isinstance(img_arr_edit, np.ndarray) is not True:
        raise TypeError
    if len(img_arr_edit.shape) != 3 or img_arr_edit.shape[2] != 3:
        raise NotImplementedError
    known_array = np.ones_like(img_arr_edit)  # create array with all values to True
    offset_x, offset_y = offset
    spacing_x, spacing_y = spacing
    if not all(isinstance(i, int) for i in [offset_x, offset_y, spacing_y, spacing_x]):
        raise ValueError
    if (0 <= offset_x <= 32) is not True or (0 <= offset_y <= 32) is not True or (2 <= spacing_x <= 8) is not True or (2 <= spacing_y <= 8) is not True:
        raise ValueError
    knows_pixels = math.ceil((img_arr_edit.shape[0] - offset_y) / spacing_y) * math.ceil((img_arr_edit.shape[1] - offset_x) / spacing_x)
    if knows_pixels < 144:
        raise ValueError
    datatype_img_arr_edit = img_arr_edit.dtype
    # create base shape of array to append new items because we cannot create a purely empty array
    target_array = np.zeros(shape=(1, 3), dtype=datatype_img_arr_edit)
    input_array = img_arr_edit.copy()
    for row_nr in range(img_arr_edit.shape[0]):
        if row_nr < offset_y:
            for pixel_column in range(img_arr_edit.shape[1]):
                input_array[row_nr][pixel_column] = [0, 0, 0]
                known_array[row_nr][pixel_column] = [0, 0, 0]
                target_array = np.append(target_array, [img_arr_edit[row_nr][pixel_column]], axis=0)
            continue
        elif (row_nr - offset_y + spacing_y) % spacing_y == 0:   # it's a row with values that we keep knowing
            for pixel_column in range(img_arr_edit.shape[1]):
                if pixel_column < offset_x:
                    input_array[row_nr][pixel_column] = [0, 0, 0]
                    known_array[row_nr][pixel_column] = [0, 0, 0]
                    target_array = np.append(target_array, [img_arr_edit[row_nr][pixel_column]], axis=0)
                    continue
                if (pixel_column - offset_x + spacing_x) % spacing_x == 0:
                    # it's a pixel with values that we keep knowing
                    continue
                else:
                    input_array[row_nr][pixel_column] = [0, 0, 0]
                    known_array[row_nr][pixel_column] = [0, 0, 0]
                    target_array = np.append(target_array, [img_arr_edit[row_nr][pixel_column]], axis=0)
        else:
            for pixel_column in range(img_arr_edit.shape[1]):
                input_array[row_nr][pixel_column] = [0, 0, 0]
                known_array[row_nr][pixel_column] = [0, 0, 0]
                target_array = np.append(target_array, [img_arr_edit[row_nr][pixel_column]], axis=0)
    target_array = target_array[1:]
    target_array = target_array.ravel('F')
    return np.transpose(input_array, (2, 0, 1)), np.transpose(known_array, (2, 0, 1)), target_array


# with open(os.path.join("unittest", "unittest_inputs_outputs.pkl"), "rb") as ufh:
#     all_inputs_outputs = pkl.load(ufh)
#     input = all_inputs_outputs['inputs'][0]
#     output = all_inputs_outputs['outputs'][0]
#
#
# print(ex4(input[0], (10, 12), (2, 3)))




