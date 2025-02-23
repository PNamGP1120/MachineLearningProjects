import numpy as np
from matplotlib import pyplot as plt

def grad(x):
    return 2*x+ 5*np.cos(x)
def cost(x):
    return x**2 + 5*np.sin(x)
def myGD1(x0, eta):
    x = [x0]
    for it in range(1000):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3: # just a small number
            break
        x.append(x_new)
    return (x, it)

x = np.linspace(-5, 5, 100)
y = np.array([cost(xi) for xi in x])
dy = np.array([grad(xi) for xi in x])

x0, it = myGD1(-4, 0.05)
print(np.round(x0,2), it)

fig, ax = plt.subplots( figsize=(10, 10))
ax.plot(x,y)
ax.scatter(x0,[cost(x0i) for x0i in x0])
plt.show()



# test = [[3, 5], [-6, 7], [-4, 8], [1, -10]]
#
# cols = int(len(test)*2 / 4)
# cols = max(1, cols)
#
# fig, ax = plt.subplots(4, cols, figsize=(10, 10))
#
# for i in range(4):
#     for j in range(int(len(test)*2/4)):
#         x0, it = myGD1(test[i][j], 0.01)
#         ax[i,j].plot(x, y)
#         ax[i,j].scatter(test[i][j], cost(test[i][j]))
#         ax[i,j].scatter(x0[-1], cost(x0[-1]))
#
# plt.tight_layout()
# plt.show()


def mySGD(x0, eta, epochs, batch_size, data_x, data_y):
    """
    Stochastic Gradient Descent (SGD)

    Args:
        x0: Initial value of x
        eta: Learning rate
        epochs: Number of epochs
        batch_size: Size of mini-batch
        data_x: Data points (x values)
        data_y: Corresponding target values (y values) - not used directly in SGD for this simple example, but needed for visualization

    Returns:
        x: List of x values during optimization
        it: Number of iterations (updates)
    """

    x = [x0]
    n_data = len(data_x)

    for epoch in range(epochs):
        # Shuffle the data indices at the beginning of each epoch
        indices = np.random.permutation(n_data)

        for i in range(0, n_data, batch_size):
            # Create mini-batch
            batch_indices = indices[i:i + batch_size]
            batch_x = data_x[batch_indices]

            # Calculate the gradient for the mini-batch (averaging the gradients of individual points)
            batch_grad = np.mean(grad(batch_x))  #  <- Key change: Gradient calculated for mini-batch

            x_new = x[-1] - eta * batch_grad
            x.append(x_new)

    return (x, (epoch + 1) * (n_data // batch_size) ) # Number of iterations is epochs * (number of batches per epoch)


x0 = -4  # Initial value
eta = 0.05  # Learning rate
epochs = 100 # Number of passes through the data
batch_size = 10 # Size of mini-batch

x_sgd, it_sgd = mySGD(x0, eta, epochs, batch_size, x, y)

print(np.round(x_sgd,2), np.round(it_sgd,2))

