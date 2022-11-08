import numpy as np

def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x) -> float:
    return sigmoid(x) * (1 - sigmoid(x))

def ReLu(x) -> float:
    return np.maximum(0, x)

def ReLu_derivative(x) -> float:
    return np.where(x > 0, 1, 0)

class Layer:
    def __init__(self, input_units: int, n_units: int):
        """
        :param input_units: Number of units in the previous layer
        :param n_units: Number of units in this layer
        """
        self._bias = np.zeros((1, n_units))
        self._weights = np.random.rand(input_units, n_units)
        self._layer_input = np.array
        self._layer_preactivation = np.array
        self._layer_activation = np.array

    def forward_step(self, layer_input: np.array) -> np.array:
        """
        :param layer_input: Input to this layer
        :return: Output of this layer
        """
        self._layer_input = layer_input
        self._layer_preactivation = (self._layer_input @ self._weights) + self._bias
        self._layer_activation = np.asarray(ReLu(self._layer_preactivation))
        return self._layer_activation

    def backward_step(self, activation_loss_gradient: np.array, learning_rate: float) -> np.array:
        """
        :param activation_loss_gradient: Gradient of the loss function with respect to the activation of this layer
        :param learning_rate: Learning rate
        :return: Gradient of the loss function with respect to the input of this layer
        """
        bias_gradient = ReLu_derivative(self._layer_preactivation) * activation_loss_gradient
        weight_gradient = self._layer_input.T @ bias_gradient

        self._bias -= learning_rate * bias_gradient
        self._weights -= learning_rate * weight_gradient

        return bias_gradient @ self._weights.T

