import numpy as np
import tensorflow as tf


class Layer:
    _bias: np.array
    _weights: np.array
    _layer_input: np.array
    _layer_preactivation: np.array
    _layer_activation: np.array

    def __init__(self, input_units: int, n_units: int):
        self._bias = np.zeros((1, n_units))
        self._weights = np.random.random(input_units * n_units).reshape([input_units, n_units])

    def forward_step(self, layer_input: np.array) -> np.array:
        self._layer_input = layer_input
        self._layer_preactivation = (self._layer_input @ self._weights) + self._bias
        self._layer_activation = np.asarray(tf.nn.relu(self._layer_preactivation))
        return self._layer_activation

    def backward_step(self, activation_loss_gradient: np.array, learning_rate: float = 0.05) -> np.array:
        bias_gradient = np.vectorize(self.activation_func_derived)(self._layer_preactivation) * activation_loss_gradient
        weight_gradient = self._layer_input.transpose() @ bias_gradient

        self._bias -= learning_rate * bias_gradient
        self._weights -= learning_rate * weight_gradient

        return bias_gradient @ self._weights.transpose()

    def activation_func_derived(self, param: float):
        return 1 if param > 0 else 0  # TODO or >=?
