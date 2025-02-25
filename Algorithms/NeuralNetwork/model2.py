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
    def __init__(self, layer_sizes, eta=1):
        """
        :param layer_sizes: Danh sách số nơ-ron ở từng lớp.
                            Ví dụ: [2, 100, 50, 3] nghĩa là 2 input, 2 lớp ẩn (100, 50 nơ-ron), 3 output.
        :param eta: Learning rate.
        """
        self.layer_sizes = layer_sizes
        self.eta = eta
        self.weights, self.biases = self._init_weights()

    def _init_weights(self):
        """ Khởi tạo trọng số và bias cho từng lớp """
        weights = []
        biases = []
        for i in range(len(self.layer_sizes) - 1):
            W = 0.01 * np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1])
            b = np.zeros(self.layer_sizes[i + 1])
            weights.append(W)
            biases.append(b)
        print(weights[0].shape, weights[1].shape, weights[2].shape, biases[0].shape,biases[1].shape,biases[2].shape )
        return weights, biases

    @staticmethod
    def _softmax_stable(Z):
        e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return e_Z / e_Z.sum(axis=1, keepdims=True)

    @staticmethod
    def _crossentropy_loss(Yhat, y):
        id0 = range(Yhat.shape[0])
        return -np.mean(np.log(Yhat[id0, y]))

    def train(self, X, y, epochs=10000):
        loss_hist = []
        for i in range(epochs):
            # === Forward Pass ===
            activations = [X]
            for W, b in zip(self.weights[:-1], self.biases[:-1]):  # Lớp ẩn

                Z = activations[-1] @ W + b
                A = np.maximum(Z, 0)  # ReLU
                activations.append(A)

            # Lớp đầu ra (softmax)
            Z_out = activations[-1] @ self.weights[-1] + self.biases[-1]
            Yhat = self._softmax_stable(Z_out)
            activations.append(Yhat)

            # === Tính loss ===
            if i % 1000 == 0:
                loss = self._crossentropy_loss(Yhat, y)
                print(f"iter {i}, loss: {loss:.6f}")
                loss_hist.append(loss)

            # === Backpropagation ===
            id0 = range(Yhat.shape[0])
            Yhat[id0, y] -= 1
            E = Yhat / X.shape[0]

            dW = []
            db = []

            # Cập nhật lớp cuối cùng
            dW.append(activations[-2].T @ E)
            db.append(np.sum(E, axis=0))

            # Cập nhật các lớp ẩn
            for j in range(len(self.layer_sizes) - 2, 0, -1):
                E = E @ self.weights[j].T
                E[activations[j] <= 0] = 0  # Gradient của ReLU
                dW.insert(0, activations[j - 1].T @ E)
                db.insert(0, np.sum(E, axis=0))

            # Cập nhật trọng số và bias
            for j in range(len(self.weights)):
                self.weights[j] -= self.eta * dW[j]
                self.biases[j] -= self.eta * db[j]

        return loss_hist

    def predict(self, X):
        """ Dự đoán nhãn lớp cho dữ liệu mới """
        A = X
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            A = np.maximum(A @ W + b, 0)  # ReLU
        Z_out = A @ self.weights[-1] + self.biases[-1]
        return np.argmax(Z_out, axis=1)


# === Tạo dữ liệu Three Spirals ===
dataset = ThreeSpiralDataset(n_points=500, n_classes=3)

# === Huấn luyện MLP với n lớp ===
layer_sizes = [2, 100, 50, 3]  # Input 2, hidden layers: 100, 50, output 3
mlp = MLPClassifier(layer_sizes, eta=0.1)

# === Chia dữ liệu Train/Test không dùng scikit-learn ===
N = dataset.X.shape[0]
indices = np.random.permutation(N)
split = int(0.8 * N)  # 80% train, 20% test

X_train, y_train = dataset.X[indices[:split]], dataset.y[indices[:split]]
X_test, y_test = dataset.X[indices[split:]], dataset.y[indices[split:]]

# print("X_train", X_train.shape, "\n", X_train, "\ny_train", y_train.shape, '\n', y_train)
# Train MLP
mlp.train(X_train, y_train, epochs=5000)

# Dự đoán và đánh giá
y_pred_train = mlp.predict(X_train)
train_acc = 100 * np.mean(y_pred_train == y_train)
print(f'Training accuracy: {train_acc:.2f}%')

y_pred_test = mlp.predict(X_test)
test_acc = 100 * np.mean(y_pred_test == y_test)
print(f'Test accuracy: {test_acc:.2f}%')

dataset.plot()

