# -*- coding: utf-8 -*-

import os
import shutil

import numpy as np
from numpy import load
from numpy import savez_compressed
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from architectures import CNN
from architectures import SkipCNN
from utils_ import plot
import utils_
from datasets import *
from pathlib import Path
from PIL import Image
import glob
from matplotlib import pyplot as plt
# from line_profiler_pycharm import profile
from datareader import datareader
from torch.optim import optimizer
import torch.optim.lr_scheduler
import torch.nn
from matplotlib.image import imread
import math


# @profile  # de-comment for time and performance analysis
def evaluate_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn, device: torch.device):
    """Function for evaluation of a model `model` on the data in `dataloader` on device `device`,
    using the specified `loss_fn` loss function"""
    model.eval()
    # We will accumulate the mean loss in variable `loss`
    loss = 0
    with torch.no_grad():  # We do not need gradients for evaluation
        # Loop over all samples in `dataloader`
        for batch in tqdm(dataloader, desc="scoring", position=0):
            input_arr = batch[0]
            known_arr = batch[1]
            offsets = batch[2]
            spacings = batch[3]
            index = batch[4]
            targets = batch[5]

            input_arr = input_arr.to(device)
            known_arr = known_arr.to(device)
            targets = targets.to(device)
            outputs = model(input_arr, known_arr, offsets, spacings)
            outputs = torch.clamp(outputs, 0, 255)  # clamping the outputs to the minimum and maximum values of inputs for better performance
            # Add the current loss, which is the mean loss over all minibatch samples
            # (unless explicitly otherwise specified when creating the loss function!)
            loss += loss_fn(outputs, targets).item()
    # Get final mean loss by dividing by the number of minibatch iterations (which
    # we summed up in the above loop)
    current_batch_size = len(dataloader)
    loss /= current_batch_size
    model.train()
    return loss


# @profile  # de-comment for time and performance analysis
def main(results_path, network_config: dict, learningrate: int, weight_decay: float,
         n_updates: int, device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    """Main function that takes hyperparameters and performs training and evaluation of model"""

    dataset = TrainingDataset()

    # Prepare a path to plot to
    plotpath = os.path.join(results_path, "plots")
    os.makedirs(plotpath, exist_ok=True)

    # Split dataset into training, validation and test set
    trainingset = torch.utils.data.Subset(
        dataset,
        indices=np.arange(int(len(dataset) * (3 / 5))))
    validationset = torch.utils.data.Subset(
        dataset,
        indices=np.arange(int(len(dataset) * (3 / 5)), int(len(dataset) * (4 / 5))))
    testset = torch.utils.data.Subset(
        dataset,
        indices=np.arange(int(len(dataset) * (4 / 5)), len(dataset)))

    # Create datasets and dataloaders with rotated targets without augmentation (for evaluation)
    trainloader = torch.utils.data.DataLoader(trainingset, batch_size=32, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(validationset, batch_size=32, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)

    # Create datasets and dataloaders with rotated targets with augmentation (for training)
    # for i, sample in enumerate(trainingset):
    #     trainingset[i] = transform_image(sample)
    
    # Define a tensorboard summary writer that writes to directory "results_path/tensorboard"
    writer = SummaryWriter(log_dir=os.path.join(results_path, "tensorboard"))
    
    # Create Network
    print(network_config)
    model = CNN(**network_config)
    model.to(device)
    
    # Get mse loss function
    mse = torch.nn.MSELoss()
    
    # Get adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate, weight_decay=weight_decay)
    
    print_stats_at = 1  # print status to tensorboard every x updates
    plot_at = 20  # plot every x updates
    validate_at = 4  # evaluate model on validation set and check for new best model every x updates
    update = 0  # current update counter
    best_validation_loss = np.inf  # best validation loss so far
    update_progress_bar = tqdm(total=n_updates, desc=f"loss: {np.nan:7.5f}", position=0)

    # Save initial model as "best" model (will be overwritten later)
    saved_model_file = os.path.join(results_path, "best_model.pt")
    torch.save(model, saved_model_file)

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)  # decomment for scheduled learningrate + scheduler.step() at end of epoch

    model.train()
    torch.set_grad_enabled(True)
    # Train until n_updates updates have been reached
    while update < n_updates:
        #
        # decomment for creating new augmented samples after each epoch
        #
        # if update != 0:
        #     trainingset = TrainingDataset(new_trainingset=True)
        #     trainloader = torch.utils.data.DataLoader(trainingset, batch_size=32, shuffle=True, num_workers=2)

            # trainingset_new = np.empty(shape=(len(trainingset), 3, 100, 100), dtype=float)
            # for i, image_path in tqdm(enumerate(files), desc="Processing files", total=len(files)):
            #     with PIL.Image.open(image_path) as img:
            #         if i < len(trainingset):
            #             transformed_image = transform_image(img)
            #             trainingset_new[i] = transformed_image
            #             trainingset = trainingset_new
        # offset = (random.randint(0, 9), random.randint(0, 9))
        # spacing = (random.randint(2, 7), random.randint(2, 7))
        # input_arr = np.empty(shape=(len(trainingset), 3, 100, 100), dtype=float)  # creating new known and target dataset
        # known_arr = np.empty(shape=(len(trainingset), 3, 100, 100), dtype=float)
        # num_removed_pixels = 10000 - math.ceil((100 - offset[1]) / spacing[1]) * math.ceil((100 - offset[0]) / spacing[0])
        # targets = np.empty(shape=(len(dataset), num_removed_pixels * 3,), dtype=float)
        # for i in range(len(trainingset)):
        #     array = np.transpose(dataset[i], (1, 2, 0))
        #     input_arr[i], known_arr[i], targets[i] = datareader(array, offset, spacing)
        #     trainingset = torch.utils.data.TensorDataset(input_arr, known_arr, targets)

        # get next samples
        for batch in tqdm(trainloader, desc=f'Training epoch {update + 1} (lr = {learningrate})'):
            input_arr = batch[0]
            known_arr = batch[1]
            offsets = batch[2]
            spacings = batch[3]
            index = batch[4]
            targets = batch[5]

            input_arr = input_arr.float().to(device)
            known_arr = known_arr.float().to(device)
            targets = targets.float().to(device)

            optimizer.zero_grad()

            # Get outputs of our network
            outputs = model(input_arr, known_arr, offsets, spacings)
            # outputs = outputs*255
            loss = mse(outputs, targets)

            loss.backward()
            optimizer.step()

            # parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)  # de-comment for checking number of gradients

            # Print current status and score
            if (update + 1) % print_stats_at == 0:
                for i, (name, param) in enumerate(model.named_parameters()):
                    writer.add_histogram(tag=f"training/gradients_{i} ({name})", values=param.grad.cpu(), global_step=update)

                writer.add_scalar(tag="training/loss", scalar_value=loss.cpu(), global_step=update)

            # Plot output
            if (update + 1) % plot_at == 0:
                utils_.plot(input_arr.detach().cpu().numpy(),
                            targets.detach().cpu().numpy(),
                            outputs.detach().cpu().numpy(),
                            plotpath, update)

            #
            # # Evaluate model on validation set
            if (update + 1) % validate_at == 0:
                val_loss = evaluate_model(model, dataloader=valloader, loss_fn=mse, device=device)
                writer.add_scalar(tag="validation/loss", scalar_value=val_loss, global_step=update)
                # Add weights and gradients as arrays to tensorboard
                for i, (name, param) in enumerate(model.named_parameters()):
                    writer.add_histogram(tag=f"validation/param_{i} ({name})", values=param.cpu(), global_step=update)

                # Save best model for early stopping
                if val_loss < best_validation_loss:
                    best_validation_loss = val_loss
                    torch.save(model, saved_model_file)

            update_progress_bar.set_description(f"loss: {loss:7.5f}", refresh=True)
            update_progress_bar.update()

        update += 1
        if update >= n_updates:
            break
        # scheduler.step()  # in addition with learningrate scheduler defined before training for scheduled learningrate

    update_progress_bar.close()
    writer.close()
    print("Finished Training!")
    # Load best model and compute score on test set
    print(f"Computing scores for best model")
    model = torch.load(saved_model_file)
    train_loss = evaluate_model(model, dataloader=trainloader, loss_fn=mse, device=device)
    val_loss = evaluate_model(model, dataloader=valloader, loss_fn=mse, device=device)
    test_loss = evaluate_model(model, dataloader=testloader, loss_fn=mse, device=device)
    
    print(f"Scores:")
    print(f"  training loss: {train_loss}")
    print(f"validation loss: {val_loss}")
    print(f"      test loss: {test_loss}")
    
    # Write result to file
    with open(os.path.join(results_path, "results.txt"), "w") as rf:
        print(f"Scores:", file=rf)
        print(f"  training loss: {train_loss}", file=rf)
        print(f"validation loss: {val_loss}", file=rf)
        print(f"      test loss: {test_loss}", file=rf)


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    config_path = os.getcwd()
    parser.add_argument('-c', '--configfile', default="working_config.json", help='file to read the config from')
    args = parser.parse_args()
    
    with open(args.configfile) as cf:
        config = json.load(cf)
    main(**config)



