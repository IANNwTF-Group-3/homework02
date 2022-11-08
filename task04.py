from typing import List, Tuple
import numpy as np

import task02
import task03


def function_to_learn(val):
    return val ** 2 - val ** 3

def trainMLP(epochs: float = 1000, verbose: bool = False) -> Tuple[task03.MLP, List[float]]:
    layers = [task02.Layer(1, 10), task02.Layer(10, 1)]
    mlp = task03.MLP(layers, 0.01)

    data_to_target: List[Tuple[float, float]] = list(
        map(lambda val: (val, function_to_learn(val)), np.random.random(100)))

    return mlp, mlp.train(epochs, data_to_target, verbose=verbose)


if __name__ == '__main__':
    trainMLP()
