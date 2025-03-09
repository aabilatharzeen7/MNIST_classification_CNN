# MNIST Classification using CNN (Coding task 1)

This project implements a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset using PyTorch


## Overview

The objective of this project is to train a CNN model on the MNIST dataset to classify handwritten digits accurately. The training process leverages early stopping to prevent overfitting and saves the best-performing model based on validation loss. Final accuracies and error metrics are computed on training, validation, and test sets.

## Project structure
````
.
├── mnist_coding_task_1.py        # Main script
├── utils.py          # Utility functions
└── data/		# MNIST dataset
````
## Installation

Install Python 3.10. Then install the required libraries:

```bash
pip install torch torchvision numpy matplotlib
```
Running the code

```bash
python mnist_coding_task_1.py
```

## Model architecture

The architecture used is as follows:
````
CnnModel(
  (conv1): Conv2d(1, 48, kernel_size=(3, 3), stride=(1, 1))		# helps learn basic features in the image
  (activation) : ReLu							# introduces non linearity
  (conv2): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1))
  (activation) : ReLU
  (max_pool2d) : max_pool2d(x, 2)					# for translational invariance
  (dropout1): Dropout(p=0.0, inplace=False)				# regularization technique
  (fc1): Linear(in_features=6912, out_features=112, bias=True)
  (activation) : ReLU
  (dropout2): Dropout(p=0.17, inplace=False)
  (fc2): Linear(in_features=112, out_features=10, bias=True)
  (soft_max) : log_softmax(x, dim=1)					# Numerically stable log probabilities
)

````
The hyperparameter tuning was done using the Optuna library and certain arguments (lr, dropouts, hidden dimensions) were set accordingly:
  
````
  args_dict = {
        'gpu': 0,
        'batch_size': 64,
        'test_batch_size': 1000,
        'epochs': 50,
        'lr': 0.43,
        'val_split': 0.2,
        'dropout1': 0,
        'dropout2': 0.17,
        'hidden_1': 48,
        'hidden_2': 48,
        'hidden_3': 112,
        'patience': 15,
        'save_model': False,
        'plot_loss': False
    }
    
  ````
Early stopping based on validation loss was incorporated in order to prevent overfitting
Using log-SoftMax and negative log likelihood loss (nll_loss) improved the performance due to numerical stability


## Results obtained

````
Train loss: 0.0025, Validation loss: 0.0526, Test loss: 0.0491
Train accuracy: 99.91, Validation accuracy: 99.07, Test accuracy: 98.96
````

## Pylint score
```9.57/10```
