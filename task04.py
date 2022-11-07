from typing import List, Tuple
import numpy as np

import task02
import task03

layers = [task02.Layer(1, 10), task02.Layer(10, 10), task02.Layer(10, 1)]
mlp = task03.MLP(layers)

data_to_target: List[Tuple[float, float]] = list(map(lambda x: (x, x ** 2 + 3), np.random.random(100)))

for i in range(1000):
    for (x, t) in data_to_target:
        result = mlp.forward_step(np.asarray([x]))
        loss = mlp.backpropagation(result, np.asarray([t]))
        print("Epoch %i - Loss: %s" % (i, loss))
