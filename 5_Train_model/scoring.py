# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 07.06.2022

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This  material,  no  matter  whether  in  printed  or  electronic  form,
may  be  used  for personal  and non-commercial educational use only.
Any reproduction of this manuscript, no matter whether as a whole or in parts,
no matter whether in printed or in electronic form, requires explicit prior
acceptance of the authors.

###############################################################################

"""
import argparse
import bz2
import gzip
import lzma
import os
import zipfile

import dill as pkl
import numpy as np
import onnx
#import onnxruntime

TEST_DATA_PATH = r"C:\Users\Leon Orou\OneDrive - Johannes Kepler Universität Linz\JKU\Programming in Python II\ML Project\5_Train_Model\test\inputs.pkl"


def load_data(file: str):
    if file.endswith(".zip"):
        # "mode" cannot be "rb", so set it manually to "r" (still need the parameter or the function invocation fails)
        # noinspection PyUnusedLocal
        def zip_open(file_, mode):
            with zipfile.ZipFile(file_, "r") as myzip:
                return myzip.open(myzip.namelist()[0])

        open_fn = zip_open
    elif file.endswith(".bz2"):
        open_fn = bz2.open
    elif file.endswith(".xz"):
        open_fn = lzma.open
    elif file.endswith(".gz"):
        open_fn = gzip.open
    else:
        open_fn = open
    with open_fn(file, "rb") as pfh:
        return pkl.load(pfh)


def rmse(predictions: list, targets: list):
    def rmse_(prediction_array: np.ndarray, target_array: np.ndarray):
        if prediction_array.shape != target_array.shape:
            raise IndexError(f"Target shape is {target_array.shape} but prediction shape is {prediction_array.shape}")
        prediction_array, target_array = np.asarray(prediction_array, np.float64), np.asarray(target_array, np.float64)
        return np.sqrt(np.mean((prediction_array - target_array) ** 2))

    # Compute RMSE for each sample
    rmses = [rmse_(prediction, target) for prediction, target in zip(predictions, targets)]
    return np.mean(rmses)


def scoring_file(prediction_file: str, target_file: str):
    """Computes the mean RMSE loss on two lists of numpy arrays stored in pickle files prediction_file and targets_file

    Computation of mean RMSE loss, as used in the challenge for exercise 5. See files "example_testset.pkl" and
    "example_submission_random.pkl" for an example test set and example targets, respectively. The real test set
    (without targets) will be available as download (see assignment sheet 2).

    Parameters
    ----------
    prediction_file: str
        File path of prediction file. Has to be a pickle file (or dill file) and contain a list of numpy arrays of dtype
        uint8, as specified in assignment sheet 2. The file can optionally be compressed, which will be automatically
        determined based on its file extension, of which the following are supported:
        > ".zip": zip compression (https://docs.python.org/3/library/zipfile.html, including the requirement of the zlib
          module: https://docs.python.org/3/library/zlib.html)
        > ".gz": gzip compression (https://docs.python.org/3/library/gzip.html, also requires the zlib module)
        > ".bz2": bzip2 compression (https://docs.python.org/3/library/bz2.html)
        > ".xz": lzma compression (https://docs.python.org/3/library/lzma.html)
        If none of these file extensions match, it is assumed to be a raw pickle file.
    target_file: str
        File path of target file. Has to be a pickle file (or dill file) and contain a list of numpy arrays of dtype
        uint8, as specified in assignment sheet 2. The file can optionally be compressed (refer to "predictions_file"
        above for more details). This file will not be available for the challenge.
    """
    # Load predictions
    predictions = load_data(prediction_file)
    if not isinstance(predictions, list):
        raise TypeError(f"Expected a list of numpy arrays as pickle file. "
                        f"Got {type(predictions)} object in pickle file instead.")
    if not all([isinstance(prediction, np.ndarray) and np.uint8 == prediction.dtype
                for prediction in predictions]):
        raise TypeError("List of predictions contains elements which are not numpy arrays of dtype uint8")

    # Load targets
    targets = load_data(target_file)
    if len(targets) != len(predictions):
        raise IndexError(f"list of targets has {len(targets)} elements "
                         f"but list of submitted predictions has {len(predictions)} elements.")

    return rmse(predictions, targets)


def make_predictions(onnx_model_rt, test_data: np.ndarray):
    n_samples = len(test_data["input_arrays"])

    # Create predictions for each sample (one by one)
    predictions = []
    for sample_i in range(n_samples):
        # Normalize input by maximal value
        input_array = test_data["input_arrays"][sample_i].astype(np.float32) / 255
        known_array = test_data["known_arrays"][sample_i].astype(np.float32)
        # Stack both inputs for the network
        input_array = np.concatenate([input_array, known_array], axis=0)
        # Pretend we have a minibatch dimension
        inputs = input_array[None]  # Adds empty dimension

        # Get outputs for network
        inputs_rt = {onnx_model_rt.get_inputs()[0].name: inputs}
        outputs = onnx_model_rt.run(None, inputs_rt)[0]  # Get first return value
        # We pretended to have a minibatch dimension -> remove this dimension
        outputs = outputs[0]
        if outputs.shape != known_array.shape:
            raise ValueError(f"Unbatched model output shape is {outputs.shape} but should be {known_array.shape}")
        # Get actual prediction from (entire) raw model output
        prediction = outputs[known_array <= 0]

        # De-normalize prediction
        prediction = prediction * 255
        # Clip the predictions to a valid range (we know our prediction values can only be in range 0-255 because of
        # uint8 datatype!)
        prediction = np.clip(prediction, a_min=0, a_max=255)
        # Challenge server wants uint8 datatype for predictions
        prediction = np.asarray(prediction, dtype=np.uint8)
        # Add prediction for sample to list
        predictions.append(prediction)

    return predictions


def scoring_model(model_file: str, test_file: str, target_file: str):
    """
    Computation of mean RMSE loss, as used in the challenge for exercise 5. The targets are loaded from the specified
    "target_file" (pickle file containing list of numpy arrays), whereas the predictions are created using the model
    stored at "model_file" using the original testset input data stored at "test_file".

    Parameters
    ----------
    model_file : str
        File path of the stored (trained) model. The model must be in ONNX format, and the model output must be the
        entire image (rather than only the predicted missing pixel values as it is the case when directly submitting
        the predictions via the pickled list of numpy arrays; see function "scoring_file"). The actual predictions are
        extracted from this entire image ouput automatically. The input to the model will be the concatenated image
        data and the known array data from the original testset input data, and the batch size is fixed to 1, i.e.,
        the input shape is (N=1, C=6, H=100, W=100). The output of the model (the entire image) is thus expected to
        be (N=1, C=3, H=100, W=100), from which the actual predictions are extracted (given the known array).
    test_file: str
        File path of the original testset input data, which is a pickle file containing a dictionary with the following
        entries: "input_arrays" (list of numpy arrays), "known_arrays" (list of numpy arrays), "offsets" (list of
        integer 2-tuples), "spacings" (list of integer 2-tuples), "sample_ids" (list of strings). The file can
        optionally be compressed, which will be automatically determined based on its file extension, of which the
        following are supported:
        > ".zip": zip compression (https://docs.python.org/3/library/zipfile.html, including the requirement of the zlib
          module: https://docs.python.org/3/library/zlib.html)
        > ".gz": gzip compression (https://docs.python.org/3/library/gzip.html, also requires the zlib module)
        > ".bz2": bzip2 compression (https://docs.python.org/3/library/bz2.html)
        > ".xz": lzma compression (https://docs.python.org/3/library/lzma.html)
        If none of these file extensions match, it is assumed to be a raw pickle file.
    target_file: str
        File path of target file. Has to be a pickle file (or dill file) and contain a list of numpy arrays of dtype
        uint8, as specified in assignment sheet 2. The file can optionally be compressed (refer to "test_file" above
        for more details). This file will not be available for the challenge.
    """
    targets = load_data(target_file)
    model = onnx.load_model(model_file)
    onnx.checker.check_model(model)
    onnx_model_rt = onnxruntime.InferenceSession(model_file)
    test_data = load_data(test_file)
    predictions = make_predictions(onnx_model_rt, test_data)
    return rmse(predictions, targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str, help="Path to submission file")
    parser.add_argument("--target", type=str, default=None, help="Path to target file")
    args = parser.parse_args()
    # Infer the type of submission: 1) reported ONNX model or 2) predictions file
    if args.submission.endswith(".onnx"):
        mse_loss = scoring_model(model_file=args.submission, test_file=TEST_DATA_PATH, target_file=args.target)
    else:
        # Prediction files are too big to keep, so ensure that they are always deleted after use
        try:
            mse_loss = scoring_file(prediction_file=args.submission, target_file=args.target)
        finally:
            if os.path.exists(args.submission):
                os.remove(args.submission)
    print(mse_loss)
