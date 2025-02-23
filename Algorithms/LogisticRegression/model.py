import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.0001, tolerance=1e-6):
        self.m = None
        self.c = None
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.loss_history = []

    def y_predict(self, X):
        return np.sign(np.dot(X, self.m) + self.c)

    def loss(self, y_predict, y):
        return -np.sum(y*np.log(y_predict) + (1-y)*np.log(1-y_predict))/len(y)

    def gradient(self, X, y, y_predict):
        error = (y_predict - y)
        dm = 2 * np.dot(X.T, error) / len(X)
        dc = 2 * np.mean(error)
        return dm, dc

    def update_parameters(self, dm, dc):
        self.m -= self.learning_rate * np.mean(dm)
        self.c -= self.learning_rate * dc

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def train(self, X, y, iterations=2000):
        self.m = np.zeros(X.shape[1])
        self.c = 0
        for i in range(iterations):
            y_predict = self.y_predict(X)
            loss = self.loss(y_predict, y)

            if i > 0 and abs(self.loss_history[-1] - cost) < self.tolerance:
                print(f"Gradient Descent hội tụ tại vòng {i + 1}")
                break

            dm, dc = self.gradient(X, y, y_predict)
            self.update_parameters(dm, dc)
            self.loss_history.append(loss)


    def predict(self, X):
        return self.sigmoid(np.dot(X, self.m) + self.c)