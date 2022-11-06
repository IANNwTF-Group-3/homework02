import numpy as np
import tensorflow as tf

class Layer:
    _bias: np.narray
    _weights: np.array
    _layer_input: np.array
    _layer_preactivation: np.array
    _layer_activation: np.array

    def __init__(self, n_units: int, input_units: int):
        self._bias = np.zeros(n_units)
        self._weights = np.random.random((input_units + 1) * n_units).reshape((input_units + 1, n_units))

    def forward_step(self) -> np.array:
        self._layer_preactivation = self._layer_input @ self._weights
        self._layer_activation = tf.nn.relu(self._layer_preactivation)
        return self._layer_activation

    def backward_step(self, dLdActivation: np.array):
        # TODO calculate gradients
