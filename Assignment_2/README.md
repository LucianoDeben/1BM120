# Assignment 2: Convolutional Neural Networks and AutoML

## Overview

This assignment involves the development of a Convolutional Neural Network (CNN) for the classification of product images taken along the production line of Wefabricate, a Dutch manufacturing company. The goal is to automate the visual inspection process by classifying the images into two categories: "Defect" and "No Defect".

The product in focus for this project is the front plate of an industry plug, created through injection molding. Defects in this process are considered discolored or damaged products, including holes, scratches, burns, or missing chunks.

The dataset used in this project is labeled set of 170 images of the product, split into a training set (136 images) and a test set (34 images).

## Contents

- `model.py`: This file contains the implementation of a CNN architecture for image classification.
- `train_test_optimize.py`: This file contains the functions for training, testing and optimizing the CNN on the training data and evaluating its performance on the test data with hyperparameter optimization. 
- `support.py`: This file contains helper functions used in the project, such as data loading.
- `WF-data/`: This directory contains the train and test data used in the project.
- `Assignment_2_Notebook.ipynb`: Jupyter Notebook which imports and uses all the necessary python files to run the CNN before and after hyperparameter optimization. Also it includes the learning curves and information on which hyperparameters are used during optimization.

## Requirements

A CUDA-enabled GPU is recommended for faster computation, but the code can also run on a CPU.

- Python 3.7+
- PyTorch 1.10.0+
- tqdm
- optuna
- matplotlib

## Instructions

The `Assignment_2_Notebook.ipynb` notebook is used to train and test the CNN on the provided image dataset and evaluate its performance. The script will output the model's learning curves and results after hyperparameter optimization.

To run the Jupyter notebook, follow these steps:

1. Click on `Assignment_2_Notebook.ipynb` to open the notebook.
1. Ensure you have a Kernel installed and selected.
6. Once the notebook is open, you can run each cell by clicking on it and then clicking the "Run" button at the top of the page, or by clicking the run all button.

Please ensure that you have installed all the necessary libraries listed in the "Requirements" section before running the notebook.

## Results

The results of the project are discussed in the `1BM120_Report_Assignment_2_Group_17.pdf` file.
