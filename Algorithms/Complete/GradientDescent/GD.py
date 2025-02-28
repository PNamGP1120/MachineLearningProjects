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
        """
        Khởi tạo thuật toán Gradient Descent.

        Tham số:
        - func (callable): Hàm mất mát cần tối ưu.
        - grad (callable): Đạo hàm của hàm mất mát.
        - eta (float): Learning rate (tốc độ học).
        - tol (float): Ngưỡng dừng thuật toán.
        - max_iter (int): Số vòng lặp tối đa.
        """
        self.func = func
        self.grad = grad
        self.eta = eta
        self.tol = tol
        self.max_iter = max_iter

    def optimize(self, x0):
        """
        Thực hiện tối ưu hóa bằng thuật toán Gradient Descent.

        Ý tưởng:
        - Từ điểm khởi tạo x0, lặp cập nhật x theo công thức x_{t+1} = x_t - eta * ∇f(x_t)
        - Dừng khi |∇f(x)| < tol hoặc đạt số vòng lặp tối đa.

        Tham số:
        - x0 (float): Điểm khởi tạo.

        Trả về:
        - x (list): Danh sách các giá trị x qua từng vòng lặp.
        - it (int): Số vòng lặp đã thực hiện.
        """
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
        """
        Thực hiện tối ưu hóa bằng thuật toán Gradient Descent đa biến.

        Ý tưởng:
        - Từ vector khởi tạo x0, lặp cập nhật x theo công thức x_{t+1} = x_t - eta * ∇f(x_t)
        - Với ∇f(x) là vector gradient, chứa các đạo hàm riêng.
        - Dừng khi ||∇f(x)|| < tol hoặc đạt số vòng lặp tối đa.

        Tham số:
        - x0 (list/array): Vector điểm khởi tạo nhiều chiều.

        Trả về:
        - x (list): Danh sách các vector x qua từng vòng lặp.
        - it (int): Số vòng lặp đã thực hiện.
        """
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
        """
        Khởi tạo thuật toán Mini-Batch Gradient Descent.

        Tham số:
        - func (callable): Hàm mất mát cần tối ưu.
        - grad (callable): Đạo hàm của hàm mất mát.
        - eta (float): Learning rate (tốc độ học).
        - batch_size (int): Kích thước của mini-batch.
        - tol (float): Ngưỡng dừng thuật toán.
        - max_iter (int): Số vòng lặp tối đa.
        """
        super().__init__(func, grad, eta, tol, max_iter)
        self.batch_size = batch_size

    def optimize(self, x0):
        """
        Thực hiện tối ưu hóa bằng thuật toán Mini-Batch Gradient Descent.

        Ý tưởng:
        - Từ điểm khởi tạo x0, tạo ngẫu nhiên một mini-batch các điểm quanh x0.
        - Tính trung bình gradient của các điểm trong mini-batch.
        - Cập nhật x theo công thức: x_{t+1} = x_t - eta * grad_est
        - Dừng khi |grad_est| < tol hoặc đạt số vòng lặp tối đa.

        Tham số:
        - x0 (float): Điểm khởi tạo.

        Trả về:
        - x (list): Danh sách các giá trị x qua từng vòng lặp.
        - it (int): Số vòng lặp đã thực hiện.
        """
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
        """
        Khởi tạo thuật toán Momentum Gradient Descent.

        Tham số:
        - func (callable): Hàm mất mát cần tối ưu.
        - grad (callable): Đạo hàm của hàm mất mát.
        - eta (float): Learning rate (tốc độ học).
        - gamma (float): Hệ số momentum (0 <= gamma < 1).
        - tol (float): Ngưỡng dừng thuật toán.
        - max_iter (int): Số vòng lặp tối đa.
        """
        super().__init__(func, grad, eta, tol, max_iter)
        self.gamma = gamma

    def optimize(self, x0):
        """
        Thực hiện tối ưu hóa bằng thuật toán Momentum Gradient Descent.

        Ý tưởng:
        - Từ điểm khởi tạo x0, sử dụng một biến vận tốc v để tích lũy momentum.
        - Cập nhật vận tốc: v_t = gamma * v_{t-1} + eta * ∇f(x_t)
        - Cập nhật x: x_{t+1} = x_t - v_t
        - Dừng khi |∇f(x)| < tol hoặc đạt số vòng lặp tối đa.

        Tham số:
        - x0 (float): Điểm khởi tạo.

        Trả về:
        - x (list): Danh sách các giá trị x qua từng vòng lặp.
        - it (int): Số vòng lặp đã thực hiện.
        """
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
        """
        Khởi tạo thuật toán Nesterov Accelerated Gradient.

        Tham số:
        - func (callable): Hàm mất mát cần tối ưu.
        - grad (callable): Đạo hàm của hàm mất mát.
        - eta (float): Learning rate (tốc độ học).
        - gamma (float): Hệ số momentum (0 <= gamma < 1).
        - tol (float): Ngưỡng dừng thuật toán.
        - max_iter (int): Số vòng lặp tối đa.
        """
        super().__init__(func, grad, eta, tol, max_iter)
        self.gamma = gamma

    def optimize(self, x0):
        """
        Thực hiện tối ưu hóa bằng thuật toán Nesterov Accelerated Gradient.

        Ý tưởng:
        - Từ điểm khởi tạo x0, sử dụng một biến vận tốc v để tích lũy momentum.
        - Tính vị trí tạm thời: x_temp = x_t - gamma * v_{t-1}
        - Tính gradient tại vị trí tạm thời: ∇f(x_temp)
        - Cập nhật vận tốc: v_t = gamma * v_{t-1} + eta * ∇f(x_temp)
        - Cập nhật x: x_{t+1} = x_t - v_t
        - Dừng khi |∇f(x_temp)| < tol hoặc đạt số vòng lặp tối đa.

        Tham số:
        - x0 (float): Điểm khởi tạo.

        Trả về:
        - x (list): Danh sách các giá trị x qua từng vòng lặp.
        - it (int): Số vòng lặp đã thực hiện.
        """
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


class StochasticGradientDescent(GradientDescent):
    """
    **Stochastic Gradient Descent (SGD) - Hạ Gradient Ngẫu Nhiên**

    Ý tưởng:
    - SGD là một biến thể của Gradient Descent, trong đó mỗi lần cập nhật tham số chỉ dựa trên một mẫu dữ liệu ngẫu nhiên thay vì toàn bộ tập dữ liệu.
    - Điều này giúp thuật toán cập nhật nhanh hơn, nhưng có thể dao động mạnh hơn so với Batch Gradient Descent.

    Toán học:
    - Cập nhật tham số theo công thức:
      x_{t+1} = x_t - eta * ∇f(x_i)
      với:
        + x_t: Giá trị hiện tại
        + eta: Learning rate (tốc độ học)
        + ∇f(x_i): Gradient tại một mẫu dữ liệu ngẫu nhiên i

    Điều kiện dừng:
    - Khi giá trị tuyệt đối của gradient nhỏ hơn `tol` hoặc số vòng lặp đạt `max_iter`.

    Tham số:
    - func (callable): Hàm mất mát cần tối ưu.
    - grad (callable): Đạo hàm của hàm mất mát.
    - data (list/array): Tập dữ liệu huấn luyện.
    - eta (float): Learning rate (tốc độ học).
    - tol (float): Ngưỡng dừng thuật toán.
    - max_iter (int): Số vòng lặp tối đa.

    Phương thức:
    - optimize(x0): Chạy thuật toán từ giá trị khởi tạo x0.
    """

    def __init__(self, func, grad, data, eta=0.1, tol=1e-3, max_iter=1000):
        """
        Khởi tạo thuật toán Stochastic Gradient Descent.

        Tham số:
        - func (callable): Hàm mất mát cần tối ưu.
        - grad (callable): Đạo hàm của hàm mất mát.
        - data (list/array): Tập dữ liệu huấn luyện.
        - eta (float): Learning rate (tốc độ học).
        - tol (float): Ngưỡng dừng thuật toán.
        - max_iter (int): Số vòng lặp tối đa.
        """
        super().__init__(func, grad, data, eta, tol, max_iter)

    def optimize(self, x0):
        """
        Thực hiện tối ưu hóa bằng thuật toán Stochastic Gradient Descent.

        Ý tưởng:
        - Ở mỗi vòng lặp, chọn một mẫu ngẫu nhiên từ tập dữ liệu.
        - Cập nhật tham số theo công thức x_{t+1} = x_t - eta * ∇f(x_i)
        - Dừng khi |∇f(x)| < tol hoặc đạt số vòng lặp tối đa.

        Tham số:
        - x0 (float): Điểm khởi tạo.

        Trả về:
        - x (list): Danh sách các giá trị x qua từng vòng lặp.
        - it (int): Số vòng lặp đã thực hiện.
        """
        x = [x0]
        for it in range(self.max_iter):
            i = np.random.randint(0, len(self.data))  # Chọn một mẫu dữ liệu ngẫu nhiên
            grad_val = self.grad(x[-1], self.data[i])
            x_new = x[-1] - self.eta * grad_val
            if abs(grad_val) < self.tol:
                break
            x.append(x_new)
        return x, it