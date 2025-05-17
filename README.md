# MNIST Digit Classification Neural Network

## Overview

This project implements a feedforward neural network to classify handwritten digits from the MNIST dataset. The network uses ReLU activation in the hidden layer, softmax for output, and is trained using gradient descent with He initialization for weights.

## Features

- Loads and preprocesses the MNIST dataset
- Trains a neural network with 784 input neurons, 64 hidden neurons, and 10 output neurons
- Implements forward and backward propagation
- Evaluates model performance on training, development, and test sets
- Visualizes predictions for individual images
- Achieves \~88.7% accuracy on the test set after 1000 iterations

## Requirements

- Python 3.x
- Libraries: `numpy`, `pandas`, `matplotlib`

## Installation and Usage

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib
   ```
3. Run the script:

   ```bash
   python mnist_neural_network.py
   ```

## Dataset

- The code expects the MNIST test dataset (`mnist_test.csv`) in the `/content/` directory.
- The dataset should have a `label` column and 784 pixel columns (28x28 images).
- Download the dataset from Kaggle or another source if needed.
