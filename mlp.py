import numpy as np
from matplotlib import pyplot as plt

x = np.random.rand(100) # inputs
t = x ** 3 - x ** 2 # targets and function to learn

def sigmoid(x: np.array) -> float:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.array) -> float:
    return sigmoid(x) * (1 - sigmoid(x))

def ReLU(x: np.array) -> float:
    return np.maximum(0, x)

def ReLU_derivative(x: np.array) -> float:
    return np.where(x > 0, 1, 0)

class Layer:
    def __init__(self, input_units: int, n_units: int) -> None:
        """
        n_units: number of units in the layer
        input_units: number of units in the previous layer
        """
        self.n_units = n_units
        self.input_units = input_units
        self.weights = np.random.rand(input_units, n_units) 
        self.biases = np.zeros(n_units)
        self.layer_input = np.array
        self.layer_preactivation = np.array
        self.layer_activation = np.array

    def forward_step(self, input: np.array) -> np.array:
        """
        input: input to the layer
        """
        self.layer_input = input
        self.layer_preactivation = np.matmul(self.layer_input, self.weights) + self.biases
        self.layer_activation = ReLU(self.layer_preactivation)
        return self.layer_activation

    def backward_step(self, activation_loss_gradient: np.array, learning_rate: float) -> np.array:
        bias_gradient = ReLU_derivative(self.layer_preactivation) * activation_loss_gradient
        weight_gradient = self.layer_input.transpose() @ bias_gradient        

        self.biases -= learning_rate * bias_gradient[0]
        self.weights -= learning_rate * weight_gradient
        return np.matmul(bias_gradient, self.weights.T)

class MLP:
    def __init__(self, layers: list) -> None:
        self.layers = layers

    def forward_step(self, input: np.array) -> np.array:
        for layer in self.layers:
            input = layer.forward_step(input) # propagate forward and update outputs every step
        return input

    def backpropagation(self, delta: np.array, learning_rate: float) -> None:
        for layer in reversed(self.layers):
            delta = layer.backward_step(delta, learning_rate) # record the error for each layer and give it to the previous layer


    def train(self, x: np.array, t: np.array, epochs: int, learning_rate: float) -> None:
        losses = []
        for epoch in range(epochs):
            for i in range(len(x)):
                output = self.forward_step(np.asarray([x[i]])) # forward with single value
                loss = np.sum(((output - t[i]) ** 2) / 2)
                losses.append(loss)
                self.backpropagation(np.asarray([output - t[i]]), learning_rate)

        # plot losses
        plt.plot(losses)
        plt.show()

if __name__ == "__main__":
    layers = [Layer(1, 10), Layer(10, 1)]
    model = MLP(layers)
    model.train(x, t, epochs=1000, learning_rate=0.05)