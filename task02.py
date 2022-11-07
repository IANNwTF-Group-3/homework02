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
        preactivation_loss_gradient = np.asarray(activation_loss_gradient * tf.nn.sigmoid(self._layer_preactivation) * (
                1 - tf.nn.sigmoid(self._layer_preactivation)))
        self._bias -= learning_rate * preactivation_loss_gradient

        weight_gradient = self._layer_input * preactivation_loss_gradient
        self._weights -= learning_rate * np.transpose(weight_gradient)

        return np.transpose(preactivation_loss_gradient * self._weights)
