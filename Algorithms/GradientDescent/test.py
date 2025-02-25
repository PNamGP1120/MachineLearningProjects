import numpy as np
import matplotlib.pyplot as plt

from Algorithms.GradientDescent.GD import GradientDescent, MiniBatchGradientDescent, MomentumGradientDescent, \
    NesterovAcceleratedGradient, MultivariableGradientDescent


def cost(x):
    """Hàm mất mát: f(x) = x^2 + 5*sin(x)."""
    return x ** 2 + 5 * np.sin(x)


def grad(x):
    """Đạo hàm của hàm mất mát: f'(x) = 2x + 5cos(x)."""
    return 2 * x + 5 * np.cos(x)


def test_gradient_descent():
    """
    Kiểm tra hiệu suất của các thuật toán Gradient Descent.

    - Chạy từng thuật toán với cùng điều kiện ban đầu.
    - Hiển thị số vòng lặp cần thiết để hội tụ.
    - Vẽ đồ thị quá trình tối ưu hóa.
    """

    # Thiết lập thông số
    x0 = -5  # Giá trị khởi tạo
    eta = 0.01  # Learning rate
    gamma = 0.9  # Hệ số Momentum
    batch_size = 100  # Kích thước batch cho Mini-Batch GD
    tol = 1e-6  # Ngưỡng hội tụ

    # Khởi tạo danh sách thuật toán
    algorithms = {
        "GD": GradientDescent(cost, grad, eta, tol),
        "Multivariable GD": MultivariableGradientDescent(cost, grad, eta, tol),
        "Mini-Batch GD": MiniBatchGradientDescent(cost, grad, eta, batch_size, tol),
        "Momentum GD": MomentumGradientDescent(cost, grad, eta, gamma, tol),
        "NAG": NesterovAcceleratedGradient(cost, grad, eta, gamma, tol)
    }

    results = {}

    # Vẽ đồ thị
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', 'D', '^', 'x']
    colors = ['b', 'g', 'r', 'c', 'm']

    for i, (name, algo) in enumerate(algorithms.items()):
        x_values, iterations = algo.optimize(x0)
        y_values = [cost(x) for x in x_values]

        results[name] = {
            "iterations": iterations,
            "x_min": x_values[-1],
            "f_min": cost(x_values[-1])
        }

        plt.plot(range(len(y_values)), y_values, label=f"{name} ({iterations} iter)",
                 marker=markers[i], color=colors[i], linestyle='-')

    # Cấu hình biểu đồ
    plt.xlabel("Số vòng lặp")
    plt.ylabel("Giá trị hàm mất mát")
    plt.title("So sánh các thuật toán Gradient Descent")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Hiển thị kết quả
    print("\n📊 **Kết quả chi tiết:**")
    print(f"{'Thuật toán':<25} {'Số vòng lặp':<15} {'x_min':<15} {'f_min':<15}")
    print("=" * 70)

    for name, res in results.items():
        print(f"{name:<25} {res['iterations']:<15} {res['x_min']:<15.6f} {res['f_min']:<15.6f}")


# Chạy kiểm thử
test_gradient_descent()
