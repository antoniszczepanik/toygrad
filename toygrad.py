import numpy as np

class MLP:

    def __init__(self,
                 sizes, # list of sizes for each of the layers
                 loss, # function taking two numpy arrays and returnign loss
                 activation, # when it is a list different activations are
                             # used for each of the layers
                 epochs=10, # TODO: is epochs number equal to iteration number?
                 momentum=0.001,
                 learning_rate=0.001,
                 batch_size=1,
                 bias=False,
                 ):
        self.sizes = sizes
        self.loss = loss
        self.activation = activation
        self.bias = bias
        self.batch_size = batch_size
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.layers = []
        for i in range(1, len(sizes)):
           self.layers.append(Layer(sizes[i-1], sizes[i]))

    def forward_pass(self, X):
        for layer in self.layers:
            X = X.dot(layer.w)
            if self.bias:
                X = np.add(X, layer.b)
            X = self.activation(X)
        return X


    def backward_pass(self, error):
        pass

    def train(self):
        pass

    def __repr__(self):
        attributes = ", ".join([f'{k}={v}' for k, v in self.__dict__.items()])
        return f"MLP({attributes})"

class Layer:
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.w = np.random.random((in_size, out_size))
        self.b = np.random.random((1, out_size))

    def __repr__(self):
        return f"Layer(in_size={self.in_size}, out_size={self.out_size})"

def sigmoid(X):
   return 1/(1+np.exp(-X))
