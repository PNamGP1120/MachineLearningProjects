import matplotlib.pyplot as plt
import numpy as np
def cost(x):
    # return (x-1)**2/2 -2
    return x**2 + 5*np.sin(x)

# Giả sử test là một danh sách có độ dài phù hợp
test = list(range(8))  # Ví dụ: test có 8 phần tử
cols = int(len(test) / 4)  # Số cột

# Đảm bảo cols >= 1
cols = max(1, cols)

# Tạo dữ liệu ví dụ
x = np.linspace(-4, 6, 100)
y = np.array([cost(xi) for xi in x])

# Tạo figure với 4 hàng và cols cột
fig, ax = plt.subplots(4, cols, figsize=(10, 10))

# Vẽ hình tại ax[0,0]
ax[0, 0].plot(x, y)
ax[0, 0].set_title("Hình tại (1,1)")

# Hiển thị đồ thị
plt.tight_layout()
plt.show()
