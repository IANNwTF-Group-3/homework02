import matplotlib.pyplot as plt
import numpy as np


def plot_function():
    x_values = np.random.random(100)

    y_values = x_values ** 3 - x_values ** 2

    # Bonus: Plot data points
    _, ax = plt.subplots()
    ax.plot(range(0, 100), x_values, label="data point input")
    ax.plot(range(0, 100), y_values, label="data point target")

    ax.set(xlabel='data point index', ylabel='data point value',
           title='Data point targets ($input^3 - input^2$)')
    plt.legend()
    ax.grid()

    plt.show()


if __name__ == '__main__':
    plot_function()
