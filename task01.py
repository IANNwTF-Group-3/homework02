import matplotlib.pyplot as plt
import numpy as np

x = np.random.random(100)

t = x ** 3 - x ** 2

# Bonus: Plot data points
fig, ax = plt.subplots()
ax.plot(range(0, 100), x, label="data point input")
ax.plot(range(0, 100), t, label="data point target")

ax.set(xlabel='data point index', ylabel='data point value',
       title='Data point targets ($input^3 - input^2$)')
plt.legend()
ax.grid()

plt.show()
