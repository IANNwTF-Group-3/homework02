import numpy as np
from matplotlib import pyplot as plt 
import random
import math
from typing import List, Tuple

# task 2.1

"""1. Randomly generate 100 numbers between 0 and 1 and save them to an
array ’x’. These are your input values.
2. Create an array ’t’. For each entry x[i] in x, calculate x[i]**3-x[i]**2
and save the results to t[i]. These are your targets."""

def build_dataset() -> Tuple[np.array]:
    # 1.
    x = np.array([random.uniform(0,1) for idx in range(100)])

    # 2.
    t = np.array([a**3 - a**2 + 1 for a in x])

    # optional
    plt.scatter(x, t, 10)
    plt.xlabel("inputs")
    plt.ylabel("targets")
    #plt.show()

    return x, t


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



class Layer:
    

    """• The constructor should accept an integer argument n_units, indi-
    cating the number of units in the layer.
    • The constructor should accept an integer argument ’input_units’,
    indicating the number of units in the preceding layer.
    • The constructor should instantiate a bias vector and a weight matrix
    of shape (n inputs, n units). Use random values for the weights and 

    -> nicht (units, inputs?) es sollte ja bei Matrix*Inputvals ein Vektor rauskommen mit Größe der n_unit
    -> Größe von Vektor = Anzahl der Zeilen in der Matrix = n_unit
    -> mach ich einfach mal so erstmal

    zeros for the biases.
    • instantiate empty attributes for layer-input, layer preactivation and
    layer activation
    """
    def __init__(self, input_units : int, n_units : int) -> None:
        self.input_units = input_units
        self.n_units = n_units
        self.bias = np.zeros(n_units)
        self.weight_matrix = np.matrix([[random.uniform(0,1) for col in range(input_units)] for row in range(n_units)])
        self.layer_input = None
        self.layer_preactivation = None
        self.layer_activation = None


    """
    2. A method called ’forward_step’, which returns each unit’s activation
    (i.e. output) using ReLu as the activation function.
    """ 
    def forward_step(self, layer_input : np.array) -> np.array:
        self.layer_input = layer_input
        print("\n\nmatrix: \n", self.weight_matrix, "\nlayer_input: \n",self.layer_input)

        output = np.matmul(self.weight_matrix, self.layer_input)
        output = np.squeeze(np.asarray(output))
        self.layer_preactivation = output - self.bias

        ReLu_np = np.vectorize(ReLu)
        output = ReLu_np(output)
        self.layer_activation = output
        print("output: ", output, "\n\n")

        return output

    # does nothing, because formulas??
    def backward_step(self, target : float, dL_dactivation : np.array = None, output_layer : bool = False) -> None:
        if output_layer:
            #print("self.layer_activation: ", self.layer_activation)
            #print("np.array([target for idx in self.input_units]): ", np.array([target for idx in range(self.n_units)]))
            a_t = self.layer_activation - np.array([target for idx in range(self.n_units)])
            #print("a_t", a_t, "\n")
        else:
            # compute dL_dactivation ?
            pass

        relu_derived_np = np.vectorize(ReLu_derived)
        relu = relu_derived_np(self.layer_preactivation)
        #print("relu",relu, "\n")
        vec_mat_mult = np.multiply(self.layer_input, self.weight_matrix)
        #print("matrix", self.weight_matrix)
        #print("self.")
        #print("vec_mat_mult", vec_mat_mult, "\n")
        dL_dW = np.matmul(np.multiply(a_t, relu), self.weight_matrix)
        #print("dldw", dL_dW)

        # formulas ?
        # where do i get dL/dactivation from the l+1 layer?
        # what is dL/dInputl good for?

        # ....

        # bias missing

        # update
        learn_rate = 0.05
        subtract = np.multiply(self.weight_matrix * learn_rate, dL_dW)
        self.weight_matrix -= subtract
        print("sub: ", subtract, "\nnew weights", self.weight_matrix)

        learn_rate = 0.05
        subtract = np.multiply(self.weight_matrix * learn_rate, dL_dW)
        self.weight_matrix -= subtract

class MLP:
    def __init__(self, layers : List[Layer]) -> None:
        self.layers = layers

    # takes an input an puts it through its whole network
    def forward_step(self, input : np.array) -> np.array:
        output = input
        for idx, layer in enumerate(self.layers):
            output = layer.forward_step(output)
        return output

    # start with output layer
    def backpropagation(self):
        pass





x, t = build_dataset()

lay = Layer(10, 1)
lay.forward_step(np.array([1,2,3,4,5,6,7,8,9,10]))
lay.backward_step(target=3, output_layer=True)

layer_list = [Layer(2,4), Layer(4,3), Layer(3,2)]
network = MLP(layer_list)
network.forward_step(np.array([3, 4]))