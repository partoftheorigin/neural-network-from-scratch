# Neural network from scratch
This is the code for making neural network from scratch.

## Overview

### Simple neural network

This is a [simple](http://computing.dcu.ie/~humphrys/Notes/Neural/single.neural.html) single layer feedforward neural network (perceptron). We use binary digits as our inputs and expect binary digits as our outputs. We'll use [backpropagation](http://neuralnetworksanddeeplearning.com/chap2.html) via gradient descent to train our network and make our prediction as accurate as possible.

### Two layer neural network

This is an implementation of a two-layer neural network. The training method is stochastic (online) [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) with momentum. It computes XOR for the given input. It uses two activation functions, one for each layer. One is a tanh function and the other is the sigmoid function. It uses [cross-entropy](http://neuralnetworksanddeeplearning.com/chap3.html) as it's loss function. This is all done in less than 100 lines of code. We're building this thing from scratch!

## Dependencies

Just NumPy

## Usage

Run ``python3 two_layer_neural_network.py`` in terminal to see it train, then predict.
