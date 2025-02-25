import numpy as np
import matplotlib.pyplot as plt

class SoftmaxRegression:
    """
    Softmax Regression classifier using gradient descent optimization.
    """

    def __init__(self, learning_rate=0.01, epochs=100, tol=1e-5, batch_size=10):
        """
        Initialize the Softmax Regression model.

        Parameters:
        learning_rate (float): Learning rate for gradient descent.
        epochs (int): Number of training iterations.
        tol (float): Tolerance for early stopping.
        batch_size (int): Number of samples per batch for mini-batch gradient descent.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tol = tol
        self.batch_size = batch_size
        self.W = None

    def softmax(self, Z):
        """
        Compute the softmax function.

        Parameters:
        Z (numpy array): Input array of shape (N, C), where N is the number of samples and C is the number of classes.

        Returns:
        numpy array: Softmax probabilities of shape (N, C).
        """
        e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Stability trick
        return e_Z / e_Z.sum(axis=1, keepdims=True)

    def compute_loss(self, X, y):
        """
        Compute the cross-entropy loss.

        Parameters:
        X (numpy array): Input feature matrix of shape (N, d), where d is the number of features.
        y (numpy array): True labels of shape (N,).

        Returns:
        float: Cross-entropy loss.
        """
        A = self.softmax(X.dot(self.W))
        id0 = np.arange(X.shape[0])
        # print(A, id0, y,-np.mean(np.log(A[id0, y])), '\n')

        return -np.mean(np.log(A[id0, y]))

    def compute_gradient(self, X, y):
        """
        Compute the gradient of the loss function with respect to weights.

        Parameters:
        X (numpy array): Input feature matrix of shape (N, d).
        y (numpy array): True labels of shape (N,).

        Returns:
        numpy array: Gradient matrix of shape (d, C).
        """
        A = self.softmax(X.dot(self.W))
        id0 = np.arange(X.shape[0])
        A[id0, y] -= 1
        return X.T.dot(A) / X.shape[0]

    def fit(self, X, y):
        """
        Train the Softmax Regression model using mini-batch gradient descent.

        Parameters:
        X (numpy array): Input feature matrix of shape (N, d).
        y (numpy array): True labels of shape (N,).

        Returns:
        list: History of loss values during training.
        """
        N, d = X.shape
        C = np.max(y) + 1  # Number of classes
        self.W = np.random.randn(d, C)
        W_old = self.W.copy()
        loss_hist = [self.compute_loss(X, y)]

        nbatches = int(np.ceil(N / self.batch_size))
        for ep in range(self.epochs):
            mix_ids = np.random.permutation(N)
            for i in range(nbatches):

                batch_ids = mix_ids[self.batch_size * i: min(self.batch_size * (i + 1), N)]
                X_batch, y_batch = X[batch_ids], y[batch_ids]
                self.W -= self.learning_rate * self.compute_gradient(X_batch, y_batch)
            loss_hist.append(self.compute_loss(X, y))

            if np.linalg.norm(self.W - W_old) / self.W.size < self.tol:
                break
            W_old = self.W.copy()

        return loss_hist

    def predict(self, X):
        """
        Predict class labels for given input data.

        Parameters:
        X (numpy array): Input feature matrix of shape (N, d).

        Returns:
        numpy array: Predicted class labels of shape (N,).
        """
        print(X.dot(self.W))
        return np.argmax(X.dot(self.W), axis=1)


def generate_data(C=5, N=500):
    """Generate synthetic dataset for testing the Softmax Regression model."""
    means = [[2, 2], [8, 3], [3, 6], [14, 2], [12, 8]]
    cov = [[1, 0], [0, 1]]
    X = np.vstack([np.random.multivariate_normal(m, cov, N) for m in means])
    y = np.hstack([[i] * N for i in range(C)])
    Xbar = np.hstack((X, np.ones((X.shape[0], 1))))
    return Xbar, y, X


def test_softmax_regression():
    """Train and evaluate the Softmax Regression model."""
    Xbar, y, X = generate_data()
    model = SoftmaxRegression(learning_rate=0.05, epochs=100, batch_size=10)
    model.fit(Xbar, y)
    y_pred = model.predict(Xbar)

    # Visualization
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b', 'c', 'm']
    C = np.max(y) + 1
    for i in range(C):
        plt.scatter(X[y == i, 0], X[y == i, 1], c=colors[i], label=f'Class {i}', alpha=0.6)

    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    for i in range(C):
        slope = -model.W[0, i] / model.W[1, i]
        intercept = -model.W[2, i] / model.W[1, i]
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, linestyle='--', label=f'Decision boundary {i}')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundaries of Softmax Regression')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_softmax_regression()

