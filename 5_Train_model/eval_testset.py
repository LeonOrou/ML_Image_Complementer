import dill
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from datasets import TestControlDataset
from architectures import CNN
import os
import torch
import pickle as pkl


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model: CNN
with open(os.path.join('results/', 'best_model.pt'), 'rb') as f:
    model = torch.load(f)

model.to(device)

testset = TestControlDataset(Dataset)
testset_loader = DataLoader(testset, shuffle=False, batch_size=1)

model.eval()

predictions: list = []

with torch.no_grad():
    for input_array, known_array, offsets, spacings, sample_id in tqdm(testset_loader):
        input_array = input_array.float().to(device)
        known_array = known_array.float().to(device)
        offsets = offsets.to(device)
        spacings = spacings.to(device)
        sample_id = sample_id.to(device)

        prediction = model(
            input_array,
            known_array,
            offsets,
            spacings
        )

        prediction_arr: np.ndarray = prediction.cpu().detach().numpy().astype(np.uint8)
        known_array_np: np.ndarray = known_array.cpu().detach().numpy().astype(np.uint8)

        target_array = prediction_arr[known_array_np == 0]

        predictions.append(target_array)

with open('predictions.pkl', 'wb') as f:
    pkl.dump(predictions, f)



