from typing import List, Tuple
import numpy as np

import task02
import task03


def function_to_learn(val):
    return val ** 2 - val ** 3


def trainMLP(epochs: float = 1000) -> Tuple[task03.MLP, List[float]]:
    layers = [task02.Layer(1, 10), task02.Layer(10, 1)]
    mlp = task03.MLP(layers, 0.01)

    data_to_target: List[Tuple[float, float]] = list(
        map(lambda val: (val, function_to_learn(val)), np.random.random(100)))

    avg_losses = list()

    # Training
    for i in range(epochs):
        avg_loss = 0.0
        for (x, t) in data_to_target:
            result = mlp.forward_step(np.asarray([x]))
            loss = mlp.backpropagation(result, np.asarray([t]))[0][0]
            avg_loss += loss
            print("Epoch %i - Loss: %s" % (i, loss))
            print("Prediction: %f, Target %f, Diff: %f" % (result, t, t - result))
        avg_losses.append(avg_loss / len(data_to_target))

    return mlp, avg_losses


if __name__ == '__main__':
    trainMLP()
