import numpy as np
import matplotlib.pyplot as plt

class ThreeSpiralDataset:
    def __init__(self, n_points=100, n_classes=3, noise=0.2):
        self.n_points = n_points
        self.n_classes = n_classes
        self.noise = noise
        self.X, self.y = self._generate_spiral()

    def _generate_spiral(self):
        X, y = [], []
        for j in range(self.n_classes):
            ix = np.arange(self.n_points)
            r = ix / self.n_points * 5
            t = ix / self.n_points * 2 * np.pi + (j * 2 * np.pi / self.n_classes)
            X.append(np.c_[r * np.sin(t), r * np.cos(t)] + self.noise * np.random.randn(self.n_points, 2))
            y.append(np.full(self.n_points, j))
        return np.vstack(X), np.hstack(y)

    def plot(self):
        plt.figure(figsize=(6, 6))
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='coolwarm', edgecolors='k')
        plt.title("Three Spirals Dataset")
        plt.show()


class MLPClassifier:
    def __init__(self, d0, d1, d2, eta=1):
        self.d0 = d0  # Input dimension
        self.d1 = d1  # Hidden layer size
        self.d2 = d2  # Output classes
        self.eta = eta  # Learning rate
        self.W1, self.b1, self.W2, self.b2 = self._init_weights()

    def _init_weights(self):
        W1 = 0.01 * np.random.randn(self.d0, self.d1)
        b1 = np.zeros(self.d1)
        W2 = 0.01 * np.random.randn(self.d1, self.d2)
        b2 = np.zeros(self.d2)
        return W1, b1, W2, b2

    @staticmethod
    def _softmax_stable(Z):
        e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return e_Z / e_Z.sum(axis=1, keepdims=True)

    @staticmethod
    def _crossentropy_loss(Yhat, y):
        id0 = range(Yhat.shape[0])
        return -np.mean(np.log(Yhat[id0, y]))

    def train(self, X, y, epochs=20000):
        loss_hist = []
        for i in range(epochs):
            # Forward pass
            Z1 = X.dot(self.W1) + self.b1
            A1 = np.maximum(Z1, 0)  # ReLU activation
            Z2 = A1.dot(self.W2) + self.b2
            Yhat = self._softmax_stable(Z2)

            # Compute loss
            if i % 1000 == 0:
                loss = self._crossentropy_loss(Yhat, y)
                print(f"iter {i}, loss: {loss:.6f}")
                loss_hist.append(loss)

            # Backpropagation
            id0 = range(Yhat.shape[0])
            Yhat[id0, y] -= 1
            E2 = Yhat / X.shape[0]

            dW2 = np.dot(A1.T, E2)
            db2 = np.sum(E2, axis=0)
            E1 = np.dot(E2, self.W2.T)
            E1[Z1 <= 0] = 0  # Gradient of ReLU

            dW1 = np.dot(X.T, E1)
            db1 = np.sum(E1, axis=0)

            # Gradient Descent update
            self.W1 -= self.eta * dW1
            self.b1 -= self.eta * db1
            self.W2 -= self.eta * dW2
            self.b2 -= self.eta * db2

        return loss_hist

    def predict(self, X):
        Z1 = X.dot(self.W1) + self.b1
        A1 = np.maximum(Z1, 0)
        Z2 = A1.dot(self.W2) + self.b2
        return np.argmax(Z2, axis=1)


# === Tạo dữ liệu Three Spirals ===
dataset = ThreeSpiralDataset(n_points=100, n_classes=3)
# dataset.plot()

# === Huấn luyện MLP ===
d0, d1, d2 = 2, 100, 3  # 2 input, 100 hidden, 3 output classes
mlp = MLPClassifier(d0, d1, d2, eta=1)

# === Chia dữ liệu Train/Test không dùng scikit-learn ===
N = dataset.X.shape[0]
indices = np.random.permutation(N)
split = int(0.8 * N)  # 80% train, 20% test

X_train, y_train = dataset.X[indices[:split]], dataset.y[indices[:split]]
X_test, y_test = dataset.X[indices[split:]], dataset.y[indices[split:]]

# Train MLP
mlp.train(X_train, y_train, epochs=5000)

# Dự đoán và đánh giá
y_pred_train = mlp.predict(X_train)
train_acc = 100 * np.mean(y_pred_train == y_train)
print(f'Training accuracy: {train_acc:.2f}%')

y_pred_test = mlp.predict(X_test)
test_acc = 100 * np.mean(y_pred_test == y_test)
print(f'Test accuracy: {test_acc:.2f}%')
