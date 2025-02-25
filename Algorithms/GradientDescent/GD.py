import numpy as np


class GradientDescent:
    """
    **Gradient Descent (GD) - Hạ Gradient cơ bản**

    Ý tưởng:
    - Gradient Descent là một thuật toán tối ưu nhằm tìm cực tiểu của một hàm số bằng cách di chuyển ngược hướng gradient của hàm số đó.

    Toán học:
    - Cập nhật tham số theo công thức:
      x_{t+1} = x_t - eta * ∇f(x_t)
      với:
        + x_t: Giá trị hiện tại
        + eta: Learning rate (tốc độ học)
        + ∇f(x_t): Gradient của hàm số tại x_t

    Điều kiện dừng:
    - Khi giá trị tuyệt đối của gradient nhỏ hơn `tol` hoặc số vòng lặp đạt `max_iter`.

    Tham số:
    - func (callable): Hàm mất mát cần tối ưu.
    - grad (callable): Đạo hàm của hàm mất mát.
    - eta (float): Learning rate (tốc độ học).
    - tol (float): Ngưỡng dừng thuật toán.
    - max_iter (int): Số vòng lặp tối đa.

    Phương thức:
    - optimize(x0): Chạy thuật toán từ giá trị khởi tạo x0.
    """

    def __init__(self, func, grad, eta=0.1, tol=1e-3, max_iter=1000):
        self.func = func
        self.grad = grad
        self.eta = eta
        self.tol = tol
        self.max_iter = max_iter

    def optimize(self, x0):
        x = [x0]
        for it in range(self.max_iter):
            x_new = x[-1] - self.eta * self.grad(x[-1])
            if abs(self.grad(x_new)) < self.tol:
                break
            x.append(x_new)
        return x, it


class MultivariableGradientDescent(GradientDescent):
    """
    **Multivariable Gradient Descent - Gradient Descent cho hàm nhiều biến**

    Ý tưởng:
    - Khi hàm số có nhiều biến, ta vẫn áp dụng Gradient Descent nhưng gradient ∇f(x) sẽ là vector đạo hàm riêng.

    Toán học:
    - x_{t+1} = x_t - eta * ∇f(x_t)
    - Với x, ∇f(x) là vector:
      x = [x1, x2, ..., xn]
      ∇f(x) = [∂f/∂x1, ∂f/∂x2, ..., ∂f/∂xn]

    Điều kiện dừng:
    - Khi độ lớn của gradient ||∇f(x)|| nhỏ hơn `tol`.

    Tham số:
    - Như GradientDescent, nhưng x0 là vector.
    """

    def optimize(self, x0):
        x = [np.array(x0)]
        for it in range(self.max_iter):
            x_new = x[-1] - self.eta * np.array(self.grad(x[-1]))
            if np.linalg.norm(self.grad(x_new)) < self.tol:
                break
            x.append(x_new)
        return x, it


class MiniBatchGradientDescent(GradientDescent):
    """
    **Mini-Batch Gradient Descent (MBGD) - Gradient Descent theo mini-batch**

    Ý tưởng:
    - Thay vì sử dụng toàn bộ dữ liệu (Batch GD) hoặc một điểm duy nhất (SGD), MBGD lấy một nhóm nhỏ dữ liệu (mini-batch) để cập nhật.

    Toán học:
    - x_{t+1} = x_t - eta * (1/m) * Σ∇f(x_i)
      với:
      + m: kích thước batch
      + x_i: một phần dữ liệu từ tập huấn luyện

    Điều kiện dừng:
    - Tương tự Gradient Descent.

    Tham số:
    - batch_size (int): Kích thước mini-batch.
    """

    def __init__(self, func, grad, eta=0.1, batch_size=10, tol=1e-3, max_iter=1000):
        super().__init__(func, grad, eta, tol, max_iter)
        self.batch_size = batch_size

    def optimize(self, x0):
        x = [x0]
        for it in range(self.max_iter):
            batch = np.random.uniform(x[-1] - 0.5, x[-1] + 0.5, self.batch_size)
            grad_est = np.mean([self.grad(xi) for xi in batch])
            x_new = x[-1] - self.eta * grad_est
            if abs(grad_est) < self.tol:
                break
            x.append(x_new)
        return x, it


class MomentumGradientDescent(GradientDescent):
    """
    **Momentum Gradient Descent - Gradient Descent có đà (Momentum)**

    Ý tưởng:
    - Tránh dao động mạnh trong quá trình tối ưu bằng cách sử dụng một biến nhớ gia tốc (momentum).

    Toán học:
    - v_t = gamma * v_{t-1} + eta * ∇f(x_t)
    - x_{t+1} = x_t - v_t
      với:
      + gamma: Hệ số momentum (thường từ 0.8 đến 0.99)
      + v_t: Vận tốc hiện tại

    Tham số:
    - gamma (float): Hệ số momentum.
    """

    def __init__(self, func, grad, eta=0.1, gamma=0.9, tol=1e-3, max_iter=1000):
        super().__init__(func, grad, eta, tol, max_iter)
        self.gamma = gamma

    def optimize(self, x0):
        x = [x0]
        v = 0
        for it in range(self.max_iter):
            grad_val = self.grad(x[-1])
            v = self.gamma * v + self.eta * grad_val
            x_new = x[-1] - v
            if abs(grad_val) < self.tol:
                break
            x.append(x_new)
        return x, it


class NesterovAcceleratedGradient(GradientDescent):
    """
    **Nesterov Accelerated Gradient (NAG) - Gradient Descent tăng tốc Nesterov**

    Ý tưởng:
    - Cải thiện Momentum GD bằng cách tính gradient tại vị trí "dự đoán trước".

    Toán học:
    - x_temp = x_t - gamma * v_{t-1}
    - v_t = gamma * v_{t-1} + eta * ∇f(x_temp)
    - x_{t+1} = x_t - v_t

    Tham số:
    - gamma (float): Hệ số momentum.
    """

    def __init__(self, func, grad, eta=0.1, gamma=0.9, tol=1e-3, max_iter=1000):
        super().__init__(func, grad, eta, tol, max_iter)
        self.gamma = gamma

    def optimize(self, x0):
        x = [x0]
        v = 0
        for it in range(self.max_iter):
            x_temp = x[-1] - self.gamma * v
            grad_temp = self.grad(x_temp)
            v = self.gamma * v + self.eta * grad_temp
            x_new = x[-1] - v
            if abs(grad_temp) < self.tol:
                break
            x.append(x_new)
        return x, it
