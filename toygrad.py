import numpy as np

EPSILON = 0.000_001

class MLP:

    def __init__(self,
                 loss,
                 layers,
                 epochs=10,
                 momentum=0.001,
                 learning_rate=0.1,
                 batch_size=64,
                 bias=False,
                 ):
        self.loss = loss()
        self.bias = bias
        self.batch_size = batch_size
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.layers = layers
        self.epochs = epochs
        # Make sure that layer sizes match
        for i in range(1, len(layers)):
            assert layers[i-1].out_size == layers[i].in_size, "Layer sizes don't match"


    def train(self, X, Y, X_test, Y_test):
        # Training and testing stats to return from this function.
        train_losses = []
        test_losses = []

        batch_number = len(X) // self.batch_size
        batch_sizes = [self.batch_size for _ in range(batch_number)]
        batch_sizes[-1] += len(X) % self.batch_size
        assert sum(batch_sizes) == len(X)
        for epoch in range(self.epochs):
            batch_start = 0
            losses_sum_epoch = 0
            for i, batch_size in enumerate(batch_sizes):
                X_batch = X[batch_start:batch_start+batch_size,:]
                Y_batch = Y[batch_start:batch_start+batch_size,:]

                mean_batch_loss = self._train_batch(X_batch, Y_batch).mean(axis=1)
                assert len(mean_batch_loss) == 1, "We should get mean of losses across the batch"
                losses_sum_epoch += mean_batch_loss[0]

                batch_start += batch_size

            mean_epoch_loss = losses_sum_epoch/len(batch_sizes)
            train_losses.append(mean_epoch_loss)
            test_loss = self.test(X_test, Y_test).mean(axis=1)[0]
            test_losses.append(test_loss)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:4} - train loss {mean_epoch_loss:5.5f}, test loss {test_loss:5.5f}")
        return {
            "train_losses": train_losses,
            "test_losses": test_losses,
        }

    def test(self, X_test, Y_test):
        """
        Get average loss across testing dataset.
        """
        loss_sum = 0
        for x, y in zip(X_test, Y_test):
            result = self.forward_pass(x)
            loss_sum += self.loss(result, y)
        return loss_sum / len(X_test)

    def _train_batch(self, X, Y):
        assert len(X) == len(Y), "Length of X and Y do not match"
        # Sum updates to weights and biases for the whole batch.
        w_update_sums = [np.zeros((l.in_size, l.out_size)) for l in self.layers]
        b_update_sums = [np.zeros((1, l.out_size)) for l in self.layers]
        loss_sum = 0
        for x, y in zip(X, Y):
            result = self.forward_pass(x)
            loss_sum += self.loss(result, y)
            w_update, b_update = self.backward_pass(x, y)
            w_update_sums = [w_sum + w_up for w_sum, w_up in zip(w_update_sums, w_update)]
            b_update_sums = [b_sum + b_up for b_sum, b_up in zip(b_update_sums, b_update)]

        for layer, w_update, b_update in zip(self.layers, w_update_sums, b_update_sums):
            layer.w -= (self.learning_rate/len(X))*w_update
            layer.b -= (self.learning_rate/len(X))*b_update

        return loss_sum / len(X)


    def forward_pass(self, X):
        for layer in self.layers:
            X = X.dot(layer.w)
            if self.bias:
                X = np.add(X, layer.b)
            layer.Z = X
            X = layer.activ_function(X)
            layer.A = X
        return X


    def backward_pass(self, x, y):
        """
        Perform a single backpropagation step to calculate gradient for weight
        and bias update at every layer.
        Returns two lists, each containing updates to weigths and biases at
        each of the layers.
        """
        w_updates = [None for _ in self.layers]
        b_updates = [None for _ in self.layers]

        last_layer = self.layers[-1]
        d_activation = last_layer.activ_function.derivative(last_layer.Z)
        d_loss = self.loss.derivative(last_layer.A, y)
        delta = d_loss @ d_activation

        b_updates[-1] = delta
        w_updates[-1] = np.dot(self.layers[-2].A.T, delta)

        for current_l in list(reversed(range(len(self.layers))))[1:]:
            next_l = current_l + 1
            previous_l = current_l - 1

            z = self.layers[current_l].Z
            d_activation = self.layers[current_l].activ_function.derivative(z)
            # The formula for calculating d_loss in other than last layers is
            # different. We use weights of the next layer, except derivative
            # of loss function.
            d_loss = np.dot(delta, self.layers[next_l].w.T) * d_activation
            delta = d_loss * d_activation
            b_updates[current_l] = delta
            # If this is the first layer then previous activation is just network
            # input.
            previous_A = np.array([x]) if current_l == 0 else self.layers[previous_l].A
            w_updates[current_l] = np.dot(previous_A.T, delta)

        return w_updates, b_updates

    def __repr__(self):
        attributes = ", ".join([f'{k}={v}' for k, v in self.__dict__.items()])
        return f"MLP({attributes})"

class Layer:
    def __init__(self, in_size, out_size, activ_function):
        self.in_size = in_size
        self.out_size = out_size
        self.activ_function = activ_function()
        self.w = np.random.uniform(low=EPSILON, high=1, size=(in_size, out_size))
        self.b = np.random.uniform(low=EPSILON, high=1, size=(1, out_size))
        self.momentum = np.zeros((in_size, out_size))
        # Weighted input to be used by backprop
        self.Z = np.zeros((1, out_size))
        # Output of a given layer to be used by backprop
        self.A = np.zeros((1, out_size))

    def __repr__(self):
        return (f"Layer(in_size={self.in_size}, out_size={self.out_size},"
                f"activation={self.activ_function.__class__.__name__})")


class Activation:
    pass

class Sigmoid(Activation):
    def __call__(self, X):
        return 1/(1+np.exp(-X))
    def derivative(self, X):
        # Derivative of sigmoid is defined using sigmoid itself.
        return self.__call__(X) * (1 - self.__call__(X))


class SoftMax(Activation):
    def __call__(self, X):
        # Shifting X by it's max makes softmax numerically stable.
        # aka. no infs even if X values are large.
        shifted_X = X - np.max(X)
        exps = np.exp(shifted_X)
        return exps / np.sum(exps, axis=1)

    def derivative(self, X):
        X = self.__call__(X)
        s = X.reshape(-1,1)
        return np.diagflat(s) - np.dot(s, s.T)


class Loss:
    pass

class SquaredError(Loss):
    def __call__(self, X, Y):
        return (X - Y)**2
    def derivative(self, X, Y):
        return 2*(X-Y)

class BinaryCrossEntropy(Loss):
    def __call__(self, X, Y):
        return -1*(Y * np.log(X) + (1 - Y) * np.log(1 - X))
    def derivative(self, X, Y):
        return (-Y)/X + ((1-Y)/(1-X))

class CategoricalCrossEntropy(Loss):
    def __call__(self, X, Y):
        return -1 * (Y*np.log(X))
    def derivative(self, X, Y):
        return X-Y
