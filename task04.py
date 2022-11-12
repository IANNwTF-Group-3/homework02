from typing import List, Tuple, Callable
import numpy as np

import task02
import task03


def x_squared_minus_x_cubed(val):
    return val ** 2 - val ** 3


def generate_data_points(function: Callable[[float], float], data_range: np.ndarray) -> List[Tuple[float, float]]:
    """
    Generates a list of tuples, which contains tuples of the data point together with
    the mapped value provided by the function.
    :param function: callable, which map a point of data_range to a float
    :param data_range: list of points to map
    :return:
    """
    return list(map(lambda val: (val, function(val)), data_range))


def train_task04() -> Tuple[task03.MLP, List[float]]:
    """
    Train an MLP with one hidden layer with 10 perceptrons. Uses 100 random data points and 1000 epochs.
    :return: MLP, List of losses
    """
    # One hidden layer with 10 perceptrons
    layers = [task02.Layer(1, 10), task02.Layer(10, 1)]
    # For better results use 1000 as training data amount
    training_data = generate_data_points(x_squared_minus_x_cubed, np.random.random(100))
    mlp = task03.MLP(layers, 0.01, task02.ReLu, task02.ReLu_derivative)

    return mlp, mlp.train(1000, training_data, verbose=True)


if __name__ == '__main__':
    train_task04()
