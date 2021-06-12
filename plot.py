import numpy as np
from matplotlib import pyplot as plt

from toygrad import Metric, get_metric_key

def plot_metric(metric: Metric, stats, title, **kwargs):
    train_key = get_metric_key(metric, "train")
    test_key = get_metric_key(metric, "test")
    epoch_number = range(1, len(stats[train_key]) + 1)
    plt.plot(epoch_number, stats[train_key], 'r-')
    plt.plot(epoch_number, stats[test_key], 'b-')
    plt.legend([train_key, test_key])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()


def plot_categorical_decisions(X, Y, mlp, title, binary=False, grid_res=0.1, **kwargs):
    x1, x2 = X[:,0], X[:,1]
    # define the x and y scale
    x1grid = np.arange(x1.min()-0.1, x1.max()+0.1, grid_res)
    x2grid = np.arange(x2.min()-0.1, x2.max()+0.1, grid_res)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    X_vis = np.hstack((r1,r2))
    Y_vis = []

    for x in X_vis:
        # Single variable is an output
        pred = mlp._forward_pass(x)
        Y_vis.append(pred[0] if binary else np.argmax(pred))

    Y_vis = np.array(Y_vis)
    # reshape the predictions back into a grid
    zz = Y_vis.reshape(xx.shape)
    colors = "RdBu" if binary else "RdYlBu"
    colors_label = "Probability of 1 class" if binary else "Class predictions"
    # plot the grid of x, y and z values as a surface
    c = plt.contourf(xx, yy, zz, cmap=colors, alpha=0.3)
    # add a legend, called a color bar
    bar = plt.colorbar(c)
    bar.set_label(colors_label)
    plt.scatter(x1, x2, c=Y if binary else np.argmax(Y, axis=1), s=5, cmap=colors)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(title)
    plt.show()

def plot_regression_decisions(X, Y, mlp, title, grid_res=0.1, **kwargs):
    Y_vis = []
    for x in X:
        Y_vis.append(mlp._forward_pass(x)[0])
    plt.scatter(X, Y, c='red', s=5, label="REAd")
    plt.scatter(X, Y_vis, c='blue', s=5)
    plt.legend(['Real outputs', 'MLP predictions'])
    plt.xlabel("Inputs")
    plt.ylabel("Resulting values")
    plt.title(title)
    plt.show()

def plot_network(mlp, **kwargs):
    weights = []
    for layer_num, layer in enumerate(mlp.layers):
        print("====================================")
        print(f"Layer {layer_num}")
        print("====================================")
        print("Weights")
        print(np.around(layer.w, 3))
        print("Biases")
        print(np.around(layer.b, 3))
