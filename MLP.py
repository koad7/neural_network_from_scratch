import numpy as np



class Perceptron:
    '''A single neuron with the sigmoid activation function.
        Attribute:   
            inputs: The number of inputs in the perceptron.
            bias: The bias of the perceptron. By default, it is 1.0.
    '''
    def __init__(self, inputs, bias=1.0):
        self.inputs = inputs
        self.weights = (np.random.rand(inputs+1) *2 ) -1
        self.bias = bias

    def set_weights(self, w_init):
        '''Set the weights of the perceptron.
            Parameter:
                w_init: The weights of the perceptron.
        '''
        self.weights = np.array(w_init)
    
    def sigmoid(self, x):
        '''The sigmoid activation function.
            Parameter:
                x: The input of the sigmoid function.
            Return:
                The output of the sigmoid function.
        '''
        return 1/(1+np.exp(-x))

    def run(self, x):
        '''Run the perceptron with the input x.
            Parameter:
                x: The input of the perceptron.
            Return:
                The output of the perceptron.
        '''
        x_sum = np.dot(np.append(x, self.bias), self.weights)
        return self.sigmoid(x_sum)


class MultiLayerPerceptron:
    '''A multi-layer perceptron with the sigmoid activation function.
        Attributes:
            layers: The number of layers in the MLP.
            network: The list of lits of neurons in the MLP.
            bias: The bias of the MLP. By default, it is 1.0.
            eta: The learning rate of the MLP. By default.
    '''
    def __init__(self, layers, bias=1.0):
        self.layers = np.array(layers, dtype=object)
        # self.neurons = neurons
        self.bias = bias
        self.network = [] # The lsit of lists of neurons in the MLP.
        self.values = [] # The list of lists of outputs of each layer.


        for i in range(len(self.layers)):
            self.values.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            if i > 0:
                for j in range(self.layers[i]):
                    self.network[i].append(Perceptron(inputs=self.layers[i-1], bias=self.bias))

        self.network = np.array([np.array(x) for x in self.network], dtype=object)
        self.values = np.array([np.array(x) for x in self.values], dtype=object)


    def set_weights(self, w_init):
        '''Set the weights of the MLP.
            Parameter:
                w_init: The weights of the MLP.
        '''
        for i in range(len(w_init)):
            for j in range(len(w_init[i])):
                self.network[i+1][j].set_weights(w_init[i][j])
    
    def printWeights(self):
        print(self.network)
        '''Print the weights of the MLP.'''
        for i in range(len(self.network)):
            for j in range(len(self.network[i])):
                print("Layer ",i+1,"Neuron ",j,self.network[i][j].weights)
        print("")

    def run(self, x):
        '''Run the MLP with the input x.
            Parameter:
                x: The input of the MLP.
            Return:
                The output of the MLP.
        '''
        x = np.array(x, dtype=object)
        self.values[0] = x
        for i in range(1, len(self.network)):
            for j in range(len(self.network[i])):
                self.values[i][j] = self.network[i][j].run(self.values[i-1])
        return self.values[-1]

            

   

# Test code
if __name__ == '__main__':

    # neuron = Perceptron(inputs=2)
    # neuron.set_weights([10, 10, -15]) # And

    # print("Gate: AND")
    # print("Input: [0, 0] Output: {}".format(neuron.run([0, 0])))
    # print("Input: [0, 1] Output: {}".format(neuron.run([0, 1])))
    # print("Input: [1, 0] Output: {}".format(neuron.run([1, 0])))
    # print("Input: [1, 1] Output: {}".format(neuron.run([1, 1])))

    # print("Gate: OR")
    # neuron.set_weights([15, 15, -10]) # Or
    # print("Input: [0, 0] Output: {}".format(neuron.run([0, 0])))
    # print("Input: [0, 1] Output: {}".format(neuron.run([0, 1])))
    # print("Input: [1, 0] Output: {}".format(neuron.run([1, 0])))
    # print("Input: [1, 1] Output: {}".format(neuron.run([1, 1])))
    mlp = MultiLayerPerceptron(layers=[2, 2, 1])
    mlp.set_weights([[[10, 10, -15], [15, 15, -10]], [[10, 10, -15]]])
    mlp.printWeights()
    print("MLP")
    print("Input: [0, 0] Output: {}".format(mlp.run([0, 0])))
    print("Input: [0, 1] Output: {}".format(mlp.run([0, 1])))
    print("Input: [1, 0] Output: {}".format(mlp.run([1, 0])))
    print("Input: [1, 1] Output: {}".format(mlp.run([1, 1])))