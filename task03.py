from typing import List, Tuple

import numpy as np
import task02


class MLP:
    def __init__(self, layers: List[task02.Layer], learning_rate: float):
        self._layers = layers
        self._learning_rate = learning_rate

    def forward_step(self, input_data: np.array) -> np.array:
        """
        :param input_data: Input to the network
        :return: Output of the network
        """
        result = input_data
        for layer in self._layers:
            result = layer.forward_step(result)
        return result

    def backpropagation(self, loss: np.array) -> np.array:
        """
        :param result: Output of the network
        :param target: Target value
        :return: Gradient of the loss function with respect to the input of the network
        """
        for layer in reversed(self._layers):
            loss = layer.backward_step(loss, self._learning_rate)

    def train(self, epochs: float, data: List[Tuple[float, float]], verbose: bool = False) -> List[float]:
        """
        :param epochs: Number of epochs to train
        :param data: List of tuples (input, target)
        :return: List of losses
        """
        avg_losses = []

        for i in range(epochs):
            avg_loss = 0.0
            for (x, t) in data:
                result = self.forward_step(np.asarray([x]))
                intermediate_loss = result - t
                mse_loss = 0.5 * (np.square(result - t))[0][0]
                self.backpropagation(intermediate_loss)
                avg_loss += mse_loss

                if verbose:
                    print("Epoch %i - Loss: %s" % (i, mse_loss))
                    print("Prediction: %f, Target %f, Diff: %f" % (result, t, t - result))

            avg_losses.append(avg_loss / len(data))

        return avg_losses

