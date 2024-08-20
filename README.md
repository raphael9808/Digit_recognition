# PyTorch MLP Example

This repository contains an implementation of a Multi-Layer Perceptron (MLP) model for classifying the MNIST dataset using PyTorch. consists of fully connected layers and is trained using Stochastic Gradient Descent (SGD) with Cross Entropy Loss.

## Project Overview

### Model Architecture:

**Input layer**: 28x28 *(flattened to 784 units)*
**Hidden layers**:
  - Fully Connected Layer with 128 units and ReLU activation
  - Fully Connected Layer with 64 units and ReLU activation
  - Output layer: Fully Connected Layer with 10 units *(for 10 classes)* and LogSoftmax activation
  - Loss Function: Cross Entropy Loss

**Optimizer**: Stochastic Gradient Descent (optim.SGD)

### Requirements
Python 3.x
PyTorch
NumPy
Matplotlib
