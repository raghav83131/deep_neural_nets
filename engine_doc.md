# `engine.py`

## Description
This module defines the `Value` class, which supports automatic differentiation for basic mathematical operations. The class allows for the construction of computational graphs and performs backpropagation to compute gradients, which are essential for optimization in machine learning algorithms.

## Class: `Value`
Represents a scalar value in a computational graph with support for automatic differentiation.

### Attributes
- `data` (float): The actual value.
- `grad` (float): The gradient of the value, initialized to 0.
- `_backward` (function): A function to propagate gradients backward through the computational graph. It helps track the computational graph for backpropagation. When you perform an operation, the resulting `Value` object records its input values as its children.
- `_prev` (set): A set of `Value` objects that are the inputs to the current `Value`.
- `_op` (str): A string representing the operation that produced the current `Value`. This is primarily for debugging and understanding the computational graph. It allows you to see which operation was applied to obtain the current value when you inspect the graph.
- `label` (str): An optional label for the value, useful for debugging. The labels are visualized in Graphviz.

### Methods

- `__init__(self, data, _children=(), _op='', label='')`
  Initializes the `Value` object.
  - `data` (float): The value.
  - `_children` (tuple): The input `Value` objects.
  - `_op` (str): The operation that created this `Value`.
  - `label` (str): An optional label for the value.

- `__repr__(self)`
  Returns a string representation of the `Value` object.
  - Returns: `str`

- `__add__(self, other)`
  Defines addition for `Value` objects and handles gradient propagation.
  - `other` (Value or float): The value to add.
  - Returns: `Value`

- `__pow__(self, other)`
  Defines exponentiation for `Value` objects and handles gradient propagation.
  - `other` (int or float): The exponent.
  - Returns: `Value`

- `__rmul__(self, other)`
  Defines reverse multiplication.
  - `other` (Value or float): The value to multiply.
  - Returns: `Value`

- `__truediv__(self, other)`
  Defines division for `Value` objects and handles gradient propagation.
  - `other` (Value or float): The divisor.
  - Returns: `Value`

- `__neg__(self)`
  Defines negation.
  - Returns: `Value`

- `__sub__(self, other)`
  Defines subtraction for `Value` objects.
  - `other` (Value or float): The value to subtract.
  - Returns: `Value`

- `__radd__(self, other)`
  Defines reverse addition.
  - `other` (Value or float): The value to add.
  - Returns: `Value`

- `__mul__(self, other)`
  Defines multiplication for `Value` objects and handles gradient propagation.
  - `other` (Value or float): The value to multiply.
  - Returns: `Value`

- `tanh(self)`
  Computes the hyperbolic tangent of the `Value` object and handles gradient propagation.
  - Returns: `Value`

- `exp(self)`
  Computes the exponential of the `Value` object and handles gradient propagation.
  - Returns: `Value`

- `backward(self)`
  Performs backpropagation to compute gradients. It builds a topological order of the computational graph and propagates gradients in reverse order.
