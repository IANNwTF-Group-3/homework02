from typing import List, Tuple
import numpy as np
from matplotlib import pyplot as plt

import task02
import task03


def function_to_learn(val):
    return val ** 2 - val ** 3


layers = [task02.Layer(1, 10), task02.Layer(10, 1)]
mlp = task03.MLP(layers, 0.01)

data_to_target: List[Tuple[float, float]] = list(map(lambda val: (val, function_to_learn(val)), np.random.random(100)))

# Training
for i in range(1000):
    for (x, t) in data_to_target:
        result = mlp.forward_step(np.asarray([x]))
        loss = mlp.backpropagation(result, np.asarray([t]))
        print("Epoch %i - Loss: %s" % (i, loss[0][0]))
        print("Prediction: %f, Target %f, Diff: %f" % (result, t, t - result))

plot_range = np.arange(0, 1, 0.001)

targets = list(map(function_to_learn, plot_range))
mlp_results = list(map(lambda val: mlp.forward_step(np.asarray([val]))[0][0], plot_range))

fig, ax = plt.subplots()
ax.plot(plot_range, targets, label="Target")
ax.plot(plot_range, mlp_results, label="MLP")

ax.set(xlabel='value', ylabel='function value',
       title='Learned function vs actual function')
plt.legend()
ax.grid()

plt.show()

print("Finished")
