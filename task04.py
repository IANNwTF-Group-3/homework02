from typing import List, Tuple
import numpy as np

import task02
import task03

layers = [task02.Layer(1, 3), task02.Layer(3, 3), task02.Layer(3, 3), task02.Layer(3, 1)]
mlp = task03.MLP(layers, 0.005)

data_to_target: List[Tuple[float, float]] = list(map(lambda val: (val, val ** 2 + 3), np.random.random(100)))

# Training
for i in range(1000):
    for (x, t) in data_to_target:
        result = mlp.forward_step(np.asarray([x]))
        loss = mlp.backpropagation(result, np.asarray([t]))
        print("Epoch %i - Loss: %s" % (i, loss[0]))
        print("Prediction: %f, Target %f, Diff: %f" % (result, t, t - result))
