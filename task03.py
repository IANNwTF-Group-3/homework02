from typing import List

import numpy as np
import task02


class MLP:
    _layers: List[task02.Layer]

    def __init__(self, layers: List[task02.Layer]):
        self._layers = layers

    # Returns result
    def forward_step(self, input_data: np.array) -> np.array:
        result = input_data
        for layer in self._layers:
            result = layer.forward_step(result)
        return result

    # Returns loss
    def backpropagation(self, result: np.array, target: np.array) -> np.array:
        loss_result_gradient = np.ones((1, 1)) * (result - target) # TODO optimize  #Old: (2 / np.shape(result)[0]) * (-expected + result)
        loss = loss_result_gradient
        for layer in reversed(self._layers):
            loss *= layer.backward_step(loss, 0.05)
        return loss
