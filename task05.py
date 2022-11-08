import numpy as np
from matplotlib import pyplot as plt

import task04


def plot_results():
    epochs = 1000

    (mlp, avg_losses) = task04.trainMLP(epochs)

    plot_range = np.arange(0, 1, 0.001)

    targets = list(map(task04.function_to_learn, plot_range))
    mlp_results = list(map(lambda val: mlp.forward_step(np.asarray([val]))[0][0], plot_range))

    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(range(epochs), avg_losses)
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
    plot_results()
