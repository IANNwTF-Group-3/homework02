from typing import List, Tuple, Callable
import numpy as np

import task02
import task03


def x_squared_minus_x_cubed(val):
    return val ** 2 - val ** 3


def generate_data_points(function: Callable[[float], float], data_range: np.ndarray) -> List[Tuple[float, float]]:
    return list(map(lambda val: (val, function(val)), data_range))

def train_task04():
    layers = [task02.Layer(1, 10), task02.Layer(10, 1)]
    training_data = generate_data_points(x_squared_minus_x_cubed, np.random.random(100))
    mlp = task03.MLP(layers, 0.01, task02.ReLu, task02.ReLu_derivative)

    return mlp, mlp.train(1000, training_data, verbose=True)

if __name__ == '__main__':
    train_task04()
