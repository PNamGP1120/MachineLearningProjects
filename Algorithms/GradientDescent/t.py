import numpy as np
import matplotlib.pyplot as plt
from GD import (
    GradientDescent,
    MultivariableGradientDescent,
    MiniBatchGradientDescent,
    MomentumGradientDescent,
    NesterovAcceleratedGradient
)


def test_gradient_descent():
    """
    Kiểm tra thuật toán Gradient Descent cơ bản trên hàm parabola f(x) = x^2.
    Hàm này có đạo hàm là f'(x) = 2x và có điểm cực tiểu là x = 0.
    """
    print("==== Kiểm tra Gradient Descent cơ bản ====")

    # Định nghĩa hàm mất mát và đạo hàm của nó
    def func(x):
        return x ** 2

    def grad(x):
        return 2 * x

    # Khởi tạo thuật toán
    gd = GradientDescent(func=func, grad=grad, eta=0.1, tol=1e-6, max_iter=100)

    # Tối ưu từ điểm x0 = 5
    x_values, iterations = gd.optimize(x0=5)

    print(f"Điểm bắt đầu: x0 = 5")
    print(f"Điểm kết thúc: x = {x_values[-1]:.6f}")
    print(f"Số vòng lặp: {iterations + 1}")
    print(f"Giá trị hàm tại điểm cuối: f(x) = {func(x_values[-1]):.6f}")

    # Vẽ đồ thị quá trình tối ưu
    plt.figure(figsize=(10, 6))
    x = np.linspace(-5, 5, 100)
    y = [func(xi) for xi in x]
    plt.plot(x, y, 'b-', label='f(x) = x^2')
    plt.plot(x_values, [func(xi) for xi in x_values], 'ro-', label='GD path')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Gradient Descent on f(x) = x^2')
    plt.legend()
    plt.grid(True)
    plt.savefig('gradient_descent_test.png')

    return x_values, iterations


def test_multivariable_gradient_descent():
    """
    Kiểm tra thuật toán Gradient Descent đa biến trên hàm
    f(x,y) = x^2 + 2y^2.
    Hàm này có đạo hàm là:
    - ∂f/∂x = 2x
    - ∂f/∂y = 4y
    và có điểm cực tiểu là (0,0).
    """
    print("\n==== Kiểm tra Gradient Descent đa biến ====")

    # Định nghĩa hàm mất mát và gradient của nó
    def func(x):
        return x[0] ** 2 + 2 * x[1] ** 2

    def grad(x):
        return np.array([2 * x[0], 4 * x[1]])

    # Khởi tạo thuật toán
    mvgd = MultivariableGradientDescent(func=func, grad=grad, eta=0.1, tol=1e-6, max_iter=100)

    # Tối ưu từ điểm (4, 3)
    x_values, iterations = mvgd.optimize(x0=[4, 3])

    print(f"Điểm bắt đầu: x0 = [4, 3]")
    print(f"Điểm kết thúc: x = [{x_values[-1][0]:.6f}, {x_values[-1][1]:.6f}]")
    print(f"Số vòng lặp: {iterations + 1}")
    print(f"Giá trị hàm tại điểm cuối: f(x) = {func(x_values[-1]):.6f}")

    # Vẽ đồ thị quá trình tối ưu (chỉ vẽ đường đi của tham số)
    plt.figure(figsize=(10, 6))
    x_path = [x[0] for x in x_values]
    y_path = [x[1] for x in x_values]
    plt.plot(x_path, y_path, 'ro-', label='GD path')
    plt.scatter(0, 0, c='blue', s=100, marker='*', label='Global minimum (0,0)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Multivariable Gradient Descent on f(x,y) = x^2 + 2y^2')
    plt.legend()
    plt.grid(True)
    plt.savefig('multivariable_gd_test.png')

    return x_values, iterations


def test_mini_batch_gradient_descent():
    """
    Kiểm tra thuật toán Mini-Batch Gradient Descent trên hàm f(x) = x^2.
    Hàm này có đạo hàm là f'(x) = 2x và có điểm cực tiểu là x = 0.
    """
    print("\n==== Kiểm tra Mini-Batch Gradient Descent ====")

    # Định nghĩa hàm mất mát và đạo hàm của nó
    def func(x):
        return x ** 2

    def grad(x):
        return 2 * x

    # Khởi tạo thuật toán
    mbgd = MiniBatchGradientDescent(func=func, grad=grad, eta=0.1, batch_size=10, tol=1e-6, max_iter=100)

    # Tối ưu từ điểm x0 = 5
    x_values, iterations = mbgd.optimize(x0=5)

    print(f"Điểm bắt đầu: x0 = 5")
    print(f"Điểm kết thúc: x = {x_values[-1]:.6f}")
    print(f"Số vòng lặp: {iterations + 1}")
    print(f"Giá trị hàm tại điểm cuối: f(x) = {func(x_values[-1]):.6f}")

    # Vẽ đồ thị quá trình tối ưu
    plt.figure(figsize=(10, 6))
    x = np.linspace(-5, 5, 100)
    y = [func(xi) for xi in x]
    plt.plot(x, y, 'b-', label='f(x) = x^2')
    plt.plot(x_values, [func(xi) for xi in x_values], 'go-', label='MBGD path')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Mini-Batch Gradient Descent on f(x) = x^2')
    plt.legend()
    plt.grid(True)
    plt.savefig('mini_batch_gd_test.png')

    return x_values, iterations


def test_momentum_gradient_descent():
    """
    Kiểm tra thuật toán Momentum Gradient Descent trên hàm có cực tiểu rất hẹp:
    f(x) = x^2 + 10*sin(x).
    """
    print("\n==== Kiểm tra Momentum Gradient Descent ====")

    # Định nghĩa hàm mất mát và đạo hàm của nó
    def func(x):
        return x ** 2 + 10 * np.sin(x)

    def grad(x):
        return 2 * x + 10 * np.cos(x)

    # Khởi tạo thuật toán
    mgd = MomentumGradientDescent(func=func, grad=grad, eta=0.01, gamma=0.9, tol=1e-6, max_iter=1000)

    # Tối ưu từ điểm x0 = 5
    x_values, iterations = mgd.optimize(x0=5)

    print(f"Điểm bắt đầu: x0 = 5")
    print(f"Điểm kết thúc: x = {x_values[-1]:.6f}")
    print(f"Số vòng lặp: {iterations + 1}")
    print(f"Giá trị hàm tại điểm cuối: f(x) = {func(x_values[-1]):.6f}")

    # So sánh với GD thông thường
    gd = GradientDescent(func=func, grad=grad, eta=0.01, tol=1e-6, max_iter=1000)
    x_values_gd, iterations_gd = gd.optimize(x0=5)

    print(f"\nSo sánh với GD thông thường:")
    print(f"GD - Điểm kết thúc: x = {x_values_gd[-1]:.6f}")
    print(f"GD - Số vòng lặp: {iterations_gd + 1}")

    # Vẽ đồ thị quá trình tối ưu
    plt.figure(figsize=(10, 6))
    x = np.linspace(-6, 6, 1000)
    y = [func(xi) for xi in x]
    plt.plot(x, y, 'b-', label='f(x) = x^2 + 10*sin(x)')
    plt.plot(x_values, [func(xi) for xi in x_values], 'mo-', label='Momentum GD path')
    plt.plot(x_values_gd, [func(xi) for xi in x_values_gd], 'co-', label='Standard GD path')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Momentum vs. Standard Gradient Descent')
    plt.legend()
    plt.grid(True)
    plt.savefig('momentum_gd_test.png')

    return x_values, iterations


def test_nesterov_accelerated_gradient():
    """
    Kiểm tra thuật toán Nesterov Accelerated Gradient trên hàm phức tạp:
    f(x) = x^4 - 4*x^2 + 5.
    Hàm này có hai điểm cực tiểu và một điểm cực đại.
    """
    print("\n==== Kiểm tra Nesterov Accelerated Gradient ====")

    # Định nghĩa hàm mất mát và đạo hàm của nó
    def func(x):
        return x ** 4 - 4 * x ** 2 + 5

    def grad(x):
        return 4 * x ** 3 - 8 * x

    # Khởi tạo thuật toán
    nag = NesterovAcceleratedGradient(func=func, grad=grad, eta=0.01, gamma=0.9, tol=1e-6, max_iter=1000)

    # Tối ưu từ điểm x0 = 0.1 (gần điểm cực đại)
    x_values, iterations = nag.optimize(x0=0.1)

    print(f"Điểm bắt đầu: x0 = 0.1")
    print(f"Điểm kết thúc: x = {x_values[-1]:.6f}")
    print(f"Số vòng lặp: {iterations + 1}")
    print(f"Giá trị hàm tại điểm cuối: f(x) = {func(x_values[-1]):.6f}")

    # So sánh với Momentum GD
    mgd = MomentumGradientDescent(func=func, grad=grad, eta=0.01, gamma=0.9, tol=1e-6, max_iter=1000)
    x_values_mgd, iterations_mgd = mgd.optimize(x0=0.1)

    print(f"\nSo sánh với Momentum GD:")
    print(f"Momentum GD - Điểm kết thúc: x = {x_values_mgd[-1]:.6f}")
    print(f"Momentum GD - Số vòng lặp: {iterations_mgd + 1}")

    # Vẽ đồ thị quá trình tối ưu
    plt.figure(figsize=(10, 6))
    x = np.linspace(-2.5, 2.5, 1000)
    y = [func(xi) for xi in x]
    plt.plot(x, y, 'b-', label='f(x) = x^4 - 4x^2 + 5')
    plt.plot(x_values, [func(xi) for xi in x_values], 'ro-', label='NAG path')
    plt.plot(x_values_mgd, [func(xi) for xi in x_values_mgd], 'go-', label='Momentum GD path')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Nesterov vs. Momentum Gradient Descent')
    plt.legend()
    plt.grid(True)
    plt.savefig('nesterov_gd_test.png')

    return x_values, iterations


def test_all_methods_comparison():
    """
    So sánh tất cả các phương pháp trên cùng một hàm để thấy sự khác biệt về tốc độ hội tụ.
    Hàm test: f(x) = 0.01*x^2 + 4*sin(x) với x0 = 5
    """
    print("\n==== So sánh tất cả các phương pháp ====")

    # Định nghĩa hàm mất mát và đạo hàm của nó
    def func(x):
        return 0.01 * x ** 2 + 4 * np.sin(x)

    def grad(x):
        return 0.02 * x + 4 * np.cos(x)

    # Khởi tạo các thuật toán
    gd = GradientDescent(func=func, grad=grad, eta=0.1, tol=1e-6, max_iter=200)
    mbgd = MiniBatchGradientDescent(func=func, grad=grad, eta=0.1, batch_size=10, tol=1e-6, max_iter=200)
    mgd = MomentumGradientDescent(func=func, grad=grad, eta=0.1, gamma=0.9, tol=1e-6, max_iter=200)
    nag = NesterovAcceleratedGradient(func=func, grad=grad, eta=0.1, gamma=0.9, tol=1e-6, max_iter=200)

    # Tối ưu từ điểm x0 = 5
    x0 = 5
    x_values_gd, iterations_gd = gd.optimize(x0=x0)
    x_values_mbgd, iterations_mbgd = mbgd.optimize(x0=x0)
    x_values_mgd, iterations_mgd = mgd.optimize(x0=x0)
    x_values_nag, iterations_nag = nag.optimize(x0=x0)

    # In kết quả
    print(
        f"GD:      {iterations_gd + 1} vòng lặp, kết quả x = {x_values_gd[-1]:.6f}, f(x) = {func(x_values_gd[-1]):.6f}")
    print(
        f"MBGD:    {iterations_mbgd + 1} vòng lặp, kết quả x = {x_values_mbgd[-1]:.6f}, f(x) = {func(x_values_mbgd[-1]):.6f}")
    print(
        f"MomentumGD: {iterations_mgd + 1} vòng lặp, kết quả x = {x_values_mgd[-1]:.6f}, f(x) = {func(x_values_mgd[-1]):.6f}")
    print(
        f"NAG:     {iterations_nag + 1} vòng lặp, kết quả x = {x_values_nag[-1]:.6f}, f(x) = {func(x_values_nag[-1]):.6f}")

    # Vẽ đồ thị giá trị hàm số theo vòng lặp
    plt.figure(figsize=(12, 8))

    # Tính giá trị hàm số theo vòng lặp
    y_gd = [func(x) for x in x_values_gd]
    y_mbgd = [func(x) for x in x_values_mbgd]
    y_mgd = [func(x) for x in x_values_mgd]
    y_nag = [func(x) for x in x_values_nag]

    # Vẽ đường giá trị hàm số
    plt.plot(range(len(y_gd)), y_gd, 'r-', label='Standard GD')
    plt.plot(range(len(y_mbgd)), y_mbgd, 'g-', label='Mini-Batch GD')
    plt.plot(range(len(y_mgd)), y_mgd, 'b-', label='Momentum GD')
    plt.plot(range(len(y_nag)), y_nag, 'm-', label='Nesterov AG')

    plt.xlabel('Iterations')
    plt.ylabel('Function Value')
    plt.title('Comparison of Optimization Methods')
    plt.legend()
    plt.grid(True)
    plt.savefig('all_methods_comparison.png')

    # Vẽ đồ thị hàm số và các đường tối ưu
    plt.figure(figsize=(12, 8))
    x = np.linspace(-10, 10, 1000)
    y = [func(xi) for xi in x]
    plt.plot(x, y, 'k-', label='f(x) = 0.01*x^2 + 4*sin(x)')
    plt.plot(x_values_gd, [func(x) for x in x_values_gd], 'ro-', label='GD')
    plt.plot(x_values_mbgd, [func(x) for x in x_values_mbgd], 'go-', label='MBGD')
    plt.plot(x_values_mgd, [func(x) for x in x_values_mgd], 'bo-', label='Momentum GD')
    plt.plot(x_values_nag, [func(x) for x in x_values_nag], 'mo-', label='NAG')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Optimization Paths Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('paths_comparison.png')

    return {
        'gd': (x_values_gd, iterations_gd),
        'mbgd': (x_values_mbgd, iterations_mbgd),
        'mgd': (x_values_mgd, iterations_mgd),
        'nag': (x_values_nag, iterations_nag)
    }


if __name__ == "__main__":
    # Chạy tất cả các test
    test_gradient_descent()
    test_multivariable_gradient_descent()
    test_mini_batch_gradient_descent()
    test_momentum_gradient_descent()
    test_nesterov_accelerated_gradient()
    test_all_methods_comparison()

    print("\nTất cả các test đã hoàn thành. Vui lòng kiểm tra các file hình ảnh được tạo ra.")