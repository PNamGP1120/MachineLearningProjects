def f(x):
  """Hàm số cần tối ưu hóa: f(x) = x^2"""
  return x**2

def df(x):
  """Đạo hàm của hàm f(x): f'(x) = 2x"""
  return 2*x

def gradient_descent(x_start, learning_rate, iterations):
  """
  Thuật toán Gradient Descent (GD)

  Args:
    x_start: Giá trị x khởi tạo
    learning_rate: Tốc độ học
    iterations: Số vòng lặp

  Returns:
    x: Giá trị x sau khi tối ưu
    path: Danh sách các giá trị x trong quá trình tối ưu (để vẽ đồ thị)
  """
  x = x_start
  path = [x]  # Lưu lại các giá trị x trên đường đi
  for i in range(iterations):
    gradient = df(x)  # Tính đạo hàm tại x
    x = x - learning_rate * gradient  # Cập nhật x
    path.append(x)
  return x, path

def stochastic_gradient_descent(x_start, learning_rate, iterations):
  """
  Thuật toán Stochastic Gradient Descent (SGD)

  Args:
    x_start: Giá trị x khởi tạo
    learning_rate: Tốc độ học
    iterations: Số vòng lặp

  Returns:
    x: Giá trị x sau khi tối ưu
    path: Danh sách các giá trị x trong quá trình tối ưu (để vẽ đồ thị)
  """
  x = x_start
  path = [x]  # Lưu lại các giá trị x trên đường đi
  for i in range(iterations):
    # Chọn ngẫu nhiên một điểm dữ liệu (trong trường hợp này, chỉ có 1 điểm)
    # Vì hàm số x^2 chỉ có 1 biến, chúng ta không cần dữ liệu
    gradient = df(x)  # Tính đạo hàm tại x
    x = x - learning_rate * gradient  # Cập nhật x
    path.append(x)
  return x, path

import matplotlib.pyplot as plt

# Khởi tạo các tham số
x_start = 5  # Giá trị x ban đầu
learning_rate = 0.1  # Tốc độ học
iterations = 100  # Số vòng lặp

# Chạy thuật toán GD
x_gd, path_gd = gradient_descent(x_start, learning_rate, iterations)
print(f"GD: x = {x_gd}, f(x) = {f(x_gd)}")

# Chạy thuật toán SGD
x_sgd, path_sgd = stochastic_gradient_descent(x_start, learning_rate, iterations)
print(f"SGD: x = {x_sgd}, f(x) = {f(x_sgd)}")

# Vẽ đồ thị
x_values = range(iterations + 1)  # Tạo danh sách các giá trị x để vẽ
plt.plot(x_values, path_gd, label="GD")
# plt.plot(x_values, path_sgd, label="SGD")
plt.xlabel("Vòng lặp")
plt.ylabel("Giá trị x")
plt.legend()
plt.title("So sánh GD và SGD")
plt.show()