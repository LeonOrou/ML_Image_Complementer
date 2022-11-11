# ML Project for predicting missing pixels in image files

Contains a Convolutional Neural Network (CNN) that is trained to predict missing pixels from images. 
The missing pixels are stand-alone seperated by an x- and y-spacing which have for the training values between 2 and inlcuding 6 pixels. 
The missing pixels are starting at an x- and y-offset which have for training values between 0 and 8 pixels.
The final accuracy measured as the root mean squared error of the predicted RGB values were around 18.42 pixels.

Usage:
```
python main.py working_config.json

```
example_project
|- architectures.py
|    Classes and functions for CNN network architectures
|    You may select 'SkipCNN' in main file as architecture but this still contains dimension errors
|- datasets.py
|    Dataset creationg classes: provide trainingset in a folder called 'training'
|    If you remove 'transform_chain_raw'  in the train indices when creating the dataset, it will perform data augmentation on the trainingset
|    uncommenting the code for the full dataset and commenting out the subset code will use all of your image files provided in the training folder
|    You may choose a random subset size for faster computation or debugging in the subset dataset code
|    You may provide an inputs.pkl test file for running on unknown data
|- main.py
|    Main file. In this case also includes training and evaluation routines
|    Uncommenting the rows right below the while training loop will create a new transformed trainingset at the beginning of every epoch (mind the computation time)
|    Will create a results folder containing test plots, the tensorboard files(e.g. for parsing in web) the best_model.pt file and the best losses in a results.txt file
|- Datareader.py
|    Reads through all files given in the dataset and creates the known_array (containing True/False if pixel is known), input_array (with missing pixels) and target_array (not needed in this build because we compare to full image in evaluation)
|- eval_testset.py
|    If inputs.pkl file for testing reasons is given in this format, run it with this file (best_model.pt is chosen)
|    Returns predictions.pkl file with all the predicted images
|- utils.py
|    Contains the plotting function for the main file
|- working_config.json
|    Containing the hyperparameters for the main function. Can also be done via command line arguments to main.py.
|    You may change your device to 'cpu' if you don't have cuda available
```
