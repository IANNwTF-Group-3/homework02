import numpy as np
from matplotlib import pyplot as plt 
import random
import math
from typing import List, Tuple

# builds the dataset according to the paper
def build_dataset() -> Tuple[np.array]:
    # 1.
    x = np.array([random.uniform(0,1) for idx in range(100)])

    # 2.
    t = np.array([a**3 - a**2 + 1 for a in x])

    # optional
    #plt.scatter(x, t, 10)
    #plt.xlabel("inputs")
    #plt.ylabel("targets")
    #plt.show()

    return x, t

# activation functions
# just the sigmoids are used, since only then the avg will approach 0
def ReLu(x : float) -> float:
    if x > 0:
        return x
    else:
        return 0

def ReLu_derived(x : float) -> float:
    if x > 0:
        return 1
    else:
        return 0

def sigmoid(x : float) -> float:
    return 1 / (1 + math.e**(-x))

def sigmoid_derived(x : float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))


# Layer class which represents one layer of the neural network
class Layer:
    
    def __init__(self, input_units : int, n_units : int) -> None:
        self.input_units = input_units
        self.n_units = n_units
        self.bias = np.zeros(n_units)
        self.weight_matrix = np.matrix([[random.uniform(0,1) for col in range(input_units)] for row in range(n_units)])
        self.layer_input = None
        self.layer_preactivation = None
        self.layer_activation = None


    # performs forward step, returns output
    def forward_step(self, layer_input : np.array) -> np.array:
        self.layer_input = layer_input
        #print("\n\nmatrix: \n", self.weight_matrix, "\nlayer_input: \n",self.layer_input)

        output = np.matmul(self.weight_matrix, self.layer_input)
        output = np.squeeze(np.asarray(output))
        self.layer_preactivation = output + self.bias

        Sigmoid_np = np.vectorize(sigmoid_derived)
        output = Sigmoid_np(output)
        self.layer_activation = output

        return output

    # perform a backwards step
    def backward_step(self, target : float, dL_dactivation : np.array = None, output_layer : bool = False) -> None:
        
        # if the current layer is the output layer, then dL_dactivation is determined directly by the loss function
        if output_layer:
            dL_dactivation = self.layer_activation - np.array([target for idx in range(self.n_units)])

        # WEIGHTS!!!
        # Formulas used are derived from the courseware, since we dont really understand everything on the paper
        # maybe they are the same, probably not
        sigmoid_derived_np = np.vectorize(sigmoid_derived)
        sigmoid = sigmoid_derived_np(self.layer_preactivation)
        dd_dW = np.multiply(self.layer_input, self.weight_matrix)
        dL_dd = np.multiply(dL_dactivation, sigmoid)
        dL_dW = np.matmul(dL_dd, self.weight_matrix) # dL_dw2 = dL/da_n * da_n/dd_n * dd_n/dw_2


        # BIASES!!!
        dL_dBl = dL_dd # used formula of homework paper this time 

        # update weights
        learn_rate = 0.05
        subtract = np.multiply(self.weight_matrix * learn_rate, dL_dW)
        self.weight_matrix -= subtract

        # update biases
        learn_rate = 0.05
        subtract = np.multiply(self.bias * learn_rate, dL_dBl)
        subtract = np.squeeze(np.asarray(subtract))
        self.bias -= subtract
        #print("\nsub: ", subtract, "\nnew biases", self.bias)

        return dL_dW

# class for representing Multiple Layers
class MLP:
    def __init__(self, layers : List[Layer]) -> None:
        self.layers = layers

    # takes an input an puts it through its whole network
    def forward_step(self, input : np.array) -> np.array:
        output = input
        for layer in self.layers:
            output = layer.forward_step(output)
        return output

    # start with output layer
    def backpropagation(self, target : float) -> None:
        for idx in range(len(self.layers)-1, -1, -1):
            if idx == len(self.layers) - 1:
                dL_dactivation = self.layers[idx].backward_step(target=target, output_layer=True)
            else:
                dL_dactivation = self.layers[idx].backward_step(target=target, dL_dactivation=dL_dactivation)


# trains the network for 1000 (10) epochs
# approaches 0 so fast, can't be correct
def train(network : MLP, x : np.array, t : np.array, EPOCHS : int = 10) -> None:
    avg_loss_plot = []
    epoch_plot = np.array([x for x in range(EPOCHS)])

    # train, save loss
    for epoch in epoch_plot:
        total_loss = 0
        for x_val, t_val in zip(x, t):
            network.forward_step(np.array([x_val]))
            network.backpropagation(np.array([t_val]))
            total_loss += network.layers[-1].layer_activation
        avg_loss_plot.append(total_loss / x.size)

    plt.plot(epoch_plot, avg_loss_plot)
    plt.show()

# build dataset, build network, train network
x, t = build_dataset()
layer_list = [Layer(1,10), Layer(10,1)]
network = MLP(layer_list)
train(network, x, t)