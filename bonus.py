import math
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt

import task02
import task03
import task04


def sin_of_1_divided_by_x(x: float):
    return (math.sin(1 / x) + 1) / 2


def train() -> Tuple[task03.MLP, List[float]]:
    """
    Trains an MLP to predict sin(1/x).
    :return: MLP, List of losses
    """
    epochs = 5000
    layer = [
        task02.Layer(1, 5),
        task02.Layer(5, 10),
        task02.Layer(10, 1)
    ]
    mlp = task03.MLP(layer, 0.01, task02.sigmoid, task02.sigmoid_derivative)

    training_data = task04.generate_data_points(sin_of_1_divided_by_x, np.random.random(1000))

    return mlp, mlp.train(epochs, training_data, True)


def visualize(mlp: task03.MLP, losses: List[float], min_x: float, max_x: float):
    """
    Visualizes the average loss per epoch and the learned function in contrast to the target function.
    :param mlp: Trained MLP
    :param losses: List of average losses per epoch
    :param min_x: Min x value for the plot
    :param max_x: Max x value for the plot
    """
    plot_range = np.arange(min_x, max_x, 0.0001)

    targets = list(map(sin_of_1_divided_by_x, plot_range))
    mlp_results = list(map(lambda val: mlp.forward_step(np.asarray([val]))[0][0], plot_range))

    _, ax = plt.subplots(ncols=2)
    ax[0].plot(range(len(losses)), losses)
    ax[0].set(xlabel='Epochs', ylabel='Average loss',
              title='Average loss per epoch')
    ax[0].grid()

    ax[1].plot(plot_range, targets, label="Target")
    ax[1].plot(plot_range, mlp_results, label="MLP")
    ax[1].set(xlabel='Value', ylabel='Function value',
              title='Learned function vs actual function')
    plt.legend()
    ax[1].grid()

    plt.show()

if __name__ == '__main__':
    mlp, losses = train()
    visualize(mlp, losses, 0.001, 1.25)
    print("Program finished.")
