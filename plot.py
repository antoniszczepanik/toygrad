import numpy as np
from matplotlib import pyplot as plt

def plot_train_test_losses(train_losses, test_losses, title):
    epoch_number = range(1, len(train_losses) + 1)
    plt.plot(epoch_number, train_losses, 'r-')
    plt.plot(epoch_number, test_losses, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()


def plot_categorical_decisions(X, Y, mlp, title, binary=False, grid_res=0.1):
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
        pred = mlp.forward_pass(x)
        Y_vis.append(pred[0][0] if binary else np.argmax(pred))

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
