# f(x) = x^2 - xin(x)
import numpy as np
import matplotlib.pyplot as plt



x = [i for i in range(-5,5,1)]
fig, ax = plt.subplots( figsize=(6, 4))
ax.plot(x, [xi**2 for xi in x])
ax.plot(x, [2*xi for xi in x])
plt.show()
#
# def gd(x0, learning):
#     global i
#     x_values = [x0]
#     for i in range(100):
#         x_new = x_values[-1] - learning * df(x_values[-1])
#         if abs(df(x_new)) < 1e-3:
#             break
#         x_values.append(x_new)
#
#     return x_values, i
#
# x = [i for i in range(-20,20)]
# x0 =[5, 10, -5, 12, -9, 15, -3, -4, -10, 13]
#
# fig, ax = plt.subplots(4,  int(np.ceil(len(x0)/4)), figsize=(12, 8))
#
# ax[0, 0].plot(x, [xi**2 for xi in x])
# plt.show()

