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


# Test code
if __name__ == '__main__':

    neuron = Perceptron(inputs=2)
    neuron.set_weights([10, 10, -15]) # And

    print("Gate: AND")
    print("Input: [0, 0] Output: {}".format(neuron.run([0, 0])))
    print("Input: [0, 1] Output: {}".format(neuron.run([0, 1])))
    print("Input: [1, 0] Output: {}".format(neuron.run([1, 0])))
    print("Input: [1, 1] Output: {}".format(neuron.run([1, 1])))

    print("Gate: OR")
    neuron.set_weights([15, 15, -10]) # Or
    print("Input: [0, 0] Output: {}".format(neuron.run([0, 0])))
    print("Input: [0, 1] Output: {}".format(neuron.run([0, 1])))
    print("Input: [1, 0] Output: {}".format(neuron.run([1, 0])))
    print("Input: [1, 1] Output: {}".format(neuron.run([1, 1])))