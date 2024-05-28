# Assignment 2: Convolutional Neural Networks and AutoML

## Overview

This assignment involves the development of a Convolutional Neural Network (CNN) for the classification of product images taken along the production line of Wefabricate, a Dutch manufacturing company. The goal is to automate the visual inspection process by classifying the images into two categories: "Defect" and "No Defect".

The product in focus for this project is the front plate of an industry plug, created through injection molding. Defects in this process are considered discolored or damaged products, including holes, scratches, burns, or missing chunks.

The dataset used in this project is a balanced and labeled set of 170 images of the product, split into a training set (136 images) and a test set (34 images).

## Contents

- `model.py`: This file contains the implementation of a CNN architecture for image classification.
- `train_and_test.py`: This file contains the code for training the CNN on the training data and evaluating its performance on the test data.
- `support.py`: This file contains helper functions used in the project, such as data loading and preprocessing functions.
- `WF-data/`: This directory contains the train and test data used in the project.

## Requirements

A CUDA-enabled GPU is recommended for faster computation, but the code can also run on a CPU.

- Python 3.7+
- PyTorch 1.10.0+
- tqdm
- optuna
- matplotlib

## Instructions

The `train_and_test.py` script is used to train the Convolutional Neural Network (CNN) on the provided image dataset and evaluate its performance. The script will output the model's performance metrics.

To execute the `train_and_test.py` script, follow these steps:

1. Open a terminal window.
2. Navigate to the directory containing the `train_and_test.py` script. You can use the `cd` command to change directories.
3. Once you're in the correct directory, run the script using the Python command followed by the script name: `python train_and_test.py`.

The script will start executing, and you should see the output.

## Results

The results of the project, including the performance metrics of the CNN, are discussed in the `1BM120_Report_Assignment_2_Group_17.pdf` file.