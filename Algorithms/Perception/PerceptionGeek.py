import numpy as np
from matplotlib import pyplot as plt




def predict(w, X):
    return np.sign(X.dot(w))
def perceptron(X, y, w_init):
    w = w_init
    while True:
        pred = predict(w, X)

        # find indexes of misclassified points
        mis_idxs = np.where(np.equal(pred, y) == False)[0]

        # number of misclassified points
        num_mis = mis_idxs.shape[0]

        if num_mis == 0: # no more misclassified points
            return w

        # random pick one misclassified point
        random_id = np.random.choice(mis_idxs, 1)[0]

        # update w
        w = w + y[random_id]*X[random_id]

means = [[-1, 0], [1, 0]]
cov = [[.3, .2], [.2, .3]]
N = 50

X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)


X = np.concatenate((X0, X1), axis = 0)
y = np.concatenate((np.ones(N), -1*np.ones(N)))
print('y_train',y, 'y_train')

Xbar = np.concatenate((np.ones((2*N, 1)), X), axis = 1)


w_init = np.random.randn(Xbar.shape[1])

w = perceptron(Xbar, y, w_init)

# Vẽ dữ liệu
plt.figure(figsize=(6, 6))
plt.scatter(X0[:, 0], X0[:, 1], color="blue", label="Class +1", edgecolors="k")
plt.scatter(X1[:, 0], X1[:, 1], color="red", label="Class -1", edgecolors="k")

# Vẽ đường quyết định: w0 + w1*x1 + w2*x2 = 0
x1_range = np.linspace(-2, 2, 100)
x2_range = -(w[0] + w[1] * x1_range) / w[2]  # Tính x2 theo công thức đường thẳng

plt.plot(x1_range, x2_range, "g--", label="Decision Boundary")  # Vẽ đường phân tách

# Cấu hình đồ thị
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Perceptron Decision Boundary")
plt.legend()
plt.grid()
plt.show()