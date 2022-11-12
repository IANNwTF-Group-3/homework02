from typing import List, Tuple, Callable

import numpy as np

import task02


class MLP:
    def __init__(self, layers: List[task02.Layer],
                 learning_rate: float,
                 activation_function: Callable[[float], np.ndarray],
                 derived_activation_function: Callable[[float], np.ndarray]):
        self._layers = layers
        self._learning_rate = learning_rate
        self._activation_function = activation_function
        self._derived_activation_function = derived_activation_function

    def forward_step(self, input_data: np.array) -> np.array:
        """
        Returns the prediction of the network to the provided input.
        :param input_data: Input to the network
        :return: Output of the network
        """
        result = input_data
        for layer in self._layers:
            result = layer.forward_step(result, self._activation_function)
        return result

    def backpropagation(self, loss: np.array) -> np.array:
        """
        Backpropagates through the MLP to update the weights and biases. Returns the loss of the network.
        This function should be called after a forward_step.
        :param loss: The gradient of the loss
        :return: Gradient of the loss function with respect to the input of the network
        """
        for layer in reversed(self._layers):
            loss = layer.backward_step(loss, self._learning_rate, self._derived_activation_function)

    def train(self, epochs: int, data: List[Tuple[float, float]], verbose: bool = False) -> List[float]:
        """
        Trains the MLP.
        :param epochs: Number of epochs to train
        :param data: List of tuples (input, target)
        :param verbose: True if the training should print additional information to the console (default: False)
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
