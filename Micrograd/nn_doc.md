# `nn.py`

## Description
This module defines classes for building and using a simple neural network with automatic differentiation support. It includes classes for a single neuron, a layer of neurons, and a multi-layer perceptron (MLP).

## Class: `Neuron`
Represents a single neuron in a neural network.

### Attributes
- `w` (list of `Value`): Weights for the inputs.
- `b` (`Value`): Bias term.

### Methods

- `__init__(self, nin)`
  Initializes the neuron with `nin` input weights and a bias.
  - `nin` (int): Number of inputs.

- `__call__(self, x)`
  Computes the output of the neuron for input `x`.
  - `x` (list of `Value`): Input values.
  - Returns: `Value`

- `parameters(self)`
  Returns the neuron's parameters (weights and bias).
  - Returns: `list of Value`

## Class: `Layer`
Represents a layer of neurons in a neural network.

### Attributes
- `neurons` (list of `Neuron`): Neurons in the layer.

### Methods

- `__init__(self, nin, nout)`
  Initializes the layer with `nin` inputs and `nout` neurons.
  - `nin` (int): Number of inputs to the layer.
  - `nout` (int): Number of neurons in the layer.

- `__call__(self, x)`
  Computes the output of the layer for input `x`.
  - `x` (list of `Value`): Input values.
  - Returns: `list of Value` or `Value`

- `parameters(self)`
  Returns the layer's parameters (weights and biases of all neurons).
  - Returns: `list of Value`

## Class: `MLP`
Represents a multi-layer perceptron (MLP) neural network.

### Attributes
- `layers` (list of `Layer`): Layers in the MLP.

### Methods

- `__init__(self, nin, nouts)`
  Initializes the MLP with a given architecture.
  - `nin` (int): Number of inputs to the network.
  - `nouts` (list of int): List of output sizes for each layer.

- `__call__(self, x)`
  Computes the output of the MLP for input `x`.
  - `x` (list of `Value`): Input values.
  - Returns: `list of Value` or `Value`

- `parameters(self)`
  Returns the MLP's parameters (weights and biases of all layers).
  - Returns: `list of Value`
