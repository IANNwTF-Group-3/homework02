from typing import List

import numpy as np
import task02


class MLP:
    _layers: List[task02.Layer]
    _learning_rate: float

    def __init__(self, layers: List[task02.Layer], learning_rate: float):
        self._layers = layers
        self._learning_rate = learning_rate

    # Returns result
    def forward_step(self, input_data: np.array) -> np.array:
        result = input_data
        for layer in self._layers:
            result = layer.forward_step(result)
        return result

    # Returns loss
    def backpropagation(self, result: np.array, target: np.array) -> np.array:
        intermediate_loss = result - target  # np.ones((1, 1)) * (result - target)  # TODO optimize  #Old: (2 / np.shape(result)[0]) * (-expected + result)
        for layer in reversed(self._layers):
            layer_loss = layer.backward_step(intermediate_loss, self._learning_rate)
            intermediate_loss = intermediate_loss * layer_loss
        return 0.5 * (result - target) ** 2
