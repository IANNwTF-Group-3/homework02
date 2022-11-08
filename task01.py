import matplotlib.pyplot as plt
import numpy as np

xValues = np.random.random(100)

yValues = xValues ** 3 - xValues ** 2

# Bonus: Plot data points
fig, ax = plt.subplots()
ax.plot(range(0, 100), xValues, label="data point input")
ax.plot(range(0, 100), yValues, label="data point target")

ax.set(xlabel='data point index', ylabel='data point value',
       title='Data point targets ($input^3 - input^2$)')
plt.legend()
ax.grid()

plt.show()
