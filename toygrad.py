from __future__ import annotations

import numpy as np

EPSILON = 0.000_001


class MLP:

    def __init__(self,
                 loss: Loss,
                 layers: list[Layer],
                 epochs=10,
                 momentum=0.001,
                 learning_rate=0.1,
                 batch_size=64,
                 bias=False,
                 # Optional list of metrics we would like to monitor
                 metrics: list[Metric] = [],
                 verbosity=1,  # 0 = no logs, 1 = every 10th epoch, 2 = all
                 ):
        self.loss = loss()
        self.bias = bias
        self.batch_size = batch_size
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.layers = layers
        self.epochs = epochs
        # Loss is also a metric so let's treat it like one.
        self.metrics = [m() for m in metrics] + [self.loss]
        for i in range(1, len(layers)):
            if (layers[i-1].out_size != layers[i].in_size):
                raise TypeError("Layer sizes don't match.")
        # Metric values are stored with epoch granularity.
        self.stats = {
            **{f"{get_metric_key(m, 'test')}": [] for m in self.metrics},
            **{f"{get_metric_key(m, 'train')}": [] for m in self.metrics}
        }
        self.verbosity = verbosity

    def train(self, X, Y, X_test, Y_test):
        """
        Model training entrypoint. Test dataset is used to calculate
        test metrics during model training.
        """
        if (X[0].shape[0] != self.layers[0].w.shape[0]):
            raise TypeError("Input shape does not match the first layer.")
        if (Y[0].shape[0] != self.layers[-1].w.shape[1]):
            raise TypeError("Output shape does not match the last layer.")
        batch_number = len(X) // self.batch_size
        batch_sizes = [self.batch_size for _ in range(batch_number)]
        batch_sizes[-1] += len(X) % self.batch_size
        for epoch in range(self.epochs):
            batch_start = 0
            batches_stats = {}
            for i, batch_size in enumerate(batch_sizes):
                X_batch = X[batch_start:batch_start+batch_size, :]
                Y_batch = Y[batch_start:batch_start+batch_size, :]

                batches_stats = self._train_batch(
                    X_batch, Y_batch, batches_stats)
                batch_start += batch_size

            # Agregate stats from multiple batches
            batch_stats_mean = {k: [float(np.mean(v))]
                                for k, v in batches_stats.items()}
            self.stats = self._update_stats(self.stats, batch_stats_mean)
            # Additionaly calcualte stats on test set.
            test_stats = self._add_stats(
                self.predict(X_test), Y_test, {}, "test")
            self.stats = self._update_stats(self.stats, test_stats)
            self._log_stats(epoch)

        return self.stats

    def predict(self, X):
        """
        Get predictions vector for the whole sample X.
        In case of clasification returns probabilites.
        """
        return np.array([self._forward_pass(x) for x in X])

    def _train_batch(self, X, Y, stats={}):
        """
        Perform forward and backwards passes on a single batch.
        It is possible to update already existing stats by passing them.
        Returns a dictionary with (updated) stats for a given batch.
        """
        # Sum updates to weights and biases for the whole batch.
        w_update_sums = [np.zeros((l.in_size, l.out_size))
                         for l in self.layers]
        b_update_sums = [np.zeros((1, l.out_size)) for l in self.layers]
        Y_hats = []
        # Global stats are with EPOCH granularity. Here we create local (batch)
        # stats dictionary.
        for x, y in zip(X, Y):
            result = self._forward_pass(x)
            Y_hats.append(result)
            w_update, b_update = self._backward_pass(x, y)
            w_update_sums = [w_sum + w_up for w_sum,
                             w_up in zip(w_update_sums, w_update)]
            b_update_sums = [b_sum + b_up for b_sum,
                             b_up in zip(b_update_sums, b_update)]

        stats = self._add_stats(np.array(Y_hats), Y, stats, "train")

        for layer, w_update, b_update in zip(self.layers, w_update_sums, b_update_sums):
            w_update = -1*(self.learning_rate/len(X))*w_update
            w_momentum = self.momentum * layer.w_prev_delta
            w_delta = w_update + w_momentum
            layer.w += w_delta

            b_update = -1*(self.learning_rate/len(X))*b_update
            b_momentum = self.momentum * layer.b_prev_delta
            b_delta = b_update + b_momentum
            layer.b += b_delta

            layer.w_prev_delta = w_delta
            layer.b_prev_delta = b_delta

        return stats

    def _forward_pass(self, x):
        """ Pass forward a single sample. """
        for layer in self.layers:
            x = x.dot(layer.w)
            x = np.add(x, layer.b)
            layer.Z = x
            x = layer.activ_function(x)
            layer.A = x
        return x[0]

    def _backward_pass(self, x, y):
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
            previous_A = np.array(
                [x]) if current_l == 0 else self.layers[previous_l].A
            w_updates[current_l] = np.dot(previous_A.T, delta)

        return w_updates, b_updates if self.bias else [0 for _ in b_updates]

    def _add_stats(self, Y_hat, Y, stats={}, m_type="test"):
        """
        Create stats dictionary with metrics calculated between Y and Y_hat.
        It is also possible to update existing stats object with new values.
        If metric has "get_std" method stds will also be added to stats.
        """
        for metric in self.metrics:
            key = get_metric_key(metric, m_type)
            if key in stats:
                stats[key].append(metric.get_value(Y_hat, Y))
            else:
                stats[key] = [metric.get_value(Y_hat, Y)]
            is_std_available = getattr(metric, "get_std", False)
            if is_std_available:
                key_std = get_metric_key(metric, f"{m_type}_std")
                if key_std in stats:
                    stats[key_std].append(metric.get_std(Y_hat, Y))
                else:
                    stats[key_std] = [metric.get_std(Y_hat, Y)]
        return stats

    def _update_stats(self, stats: dict, new_stats: dict):
        """
        Update stats dictionary with values from new_stats.
        """
        for k, v in new_stats.items():
            if k in stats:
                stats[k].extend(v)
            else:
                stats[k] = v
        return stats

    def _log_stats(self, epoch):
        def print_stats():
            info = f"Epoch {epoch+1:4}"
            for m in self.metrics:
                keys = [
                    get_metric_key(m, "train"),
                    get_metric_key(m, "train_std"),
                    get_metric_key(m, "test"),
                    get_metric_key(m, "test_std")
                ]
                for key in keys:
                    if key in self.stats:
                        info += f"\n {key}: {self.stats[key][-1]:5.3f}"
            print(info)

        if self.verbosity == 1 and (epoch+1) % 10 == 0:
            print_stats()
        elif self.verbosity == 2:
            print_stats()

    def _get_last_stat(self, metric: Metric, m_type):
        key = get_metric_key(metric, m_type)
        if key in self.stats:
            if len(self.stats[key]) > 0:
                return self.stats[key][-1]
        return np.nan

    def __str__(self):
        attributes = ", ".join([f'{k}={v}' for k, v in self.__dict__.items()])
        return f"MLP({attributes})"


class Layer:
    def __init__(self, in_size, out_size, activ_function):
        self.in_size: int = in_size
        self.out_size: int = out_size
        self.activ_function: Activation = activ_function()
        self.w = np.random.uniform(
            low=EPSILON, high=1, size=(in_size, out_size))
        self.b = np.random.uniform(low=EPSILON, high=1, size=(1, out_size))
        # Previous deltas stored to calculate momentum
        self.w_prev_delta = np.zeros((in_size, out_size))
        self.b_prev_delta = np.zeros((1, out_size))
        # Weighted input to be used by backprop
        self.Z = np.zeros((1, out_size))
        # Activation of a given layer to be used by backprop
        self.A = np.zeros((1, out_size))

    def __repr__(self):
        return (f"Layer(in_size={self.in_size}, out_size={self.out_size},"
                f"activation={self.activ_function.__class__.__name__})")


class Activation:
    def __str__(self):
        return f"{self.__class__.__name__}"


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
        result = exps / np.sum(exps, axis=1)
        # Result needs to be larger then 0 to allow exp to work properly on
        # them.
        result[result == 0] = EPSILON
        return result

    def derivative(self, X):
        X = self.__call__(X)
        s = X.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)


class Linear(Activation):
    def __call__(self, X):
        return X

    def derivative(self, X):
        return np.ones(X.shape)


class ReLU(Activation):
    def __call__(self, X):
        return np.maximum(EPSILON, X)

    def derivative(self, X):
        X = (X > 0) * 1
        return  np.maximum(EPSILON, X)


class TanH(Activation):
    def __call__(self, X):
        return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

    def derivative(self, X):
        return 1 - (self.__call__(X)**2)


class Metric:
    def __str__(self):
        return f"{self.__class__.__name__}"


class Accuracy(Metric):
    def get_value(self, Y_hat, Y):
        correct_count = 0
        for y_hat, y in zip(Y_hat, Y):
            correct_count += int(y[np.argmax(y_hat)] == 1)
        return correct_count/len(Y)


class Loss(Metric):
    def __str__(self):
        return f"{self.__class__.__name__}"

    def get_std(self, Y_hat, Y):
        return np.std(self.__call__(Y_hat, Y))

    def get_value(self, Y_hat, Y):
        return np.mean(self.__call__(Y_hat, Y))


class SquaredError(Loss):
    def __call__(self, Y_hat, Y):
        return np.multiply(Y - Y_hat, Y - Y_hat)

    def derivative(self, Y_hat, Y):
        return -2*(Y_hat-Y)


class AbsoluteError(Loss):
    def __call__(self, X, Y):
        return np.abs(X - Y)

    def derivative(self, X, Y):
        return (X-Y)


class BinaryCrossEntropy(Loss):
    def __call__(self, Y_hat, Y):
        return -1*(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))

    def derivative(self, Y_hat, Y):
        return (-Y)/Y_hat + ((1-Y)/(1-Y_hat))


class CategoricalCrossEntropy(Loss):
    def __call__(self, Y_hat, Y):
        return -1 * (Y*np.log(Y_hat))

    def derivative(self, Y_hat, Y):
        return Y_hat-Y


def get_metric_key(metric: Metric, m_type: str):
    return f"{metric}_{m_type}"
