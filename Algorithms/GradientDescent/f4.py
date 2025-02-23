import numpy as np
import matplotlib.pyplot as plt

# Define the cost function and its gradient
def cost(x):
    return x**2 + 5*np.sin(x)

def grad(x):
    return 2*x + 5*np.cos(x)

# Gradient Descent (GD)
def gradient_descent(x0, eta, iterations):
    x = [x0]
    for _ in range(iterations):
        x_new = x[-1] - eta * grad(x[-1])
        x.append(x_new)
    return x

# Stochastic Gradient Descent (SGD)
def stochastic_gradient_descent(x0, eta, iterations, data):
    x = [x0]
    for _ in range(iterations):
        # Pick a random data point (we'll just use x values for this simple example)
        i = np.random.randint(len(data))
        gradient = grad(data[i])
        x_new = x[-1] - eta * gradient
        x.append(x_new)
    return x

# Generate some sample data
x_data = np.linspace(2, 5, 100)  # For plotting the cost function
y_data = cost(x_data)

# Initial parameters
x0 = 4.5  # Starting point
eta = 0.01  # Learning rate
iterations = 50  # Number of iterations

# Run GD and SGD
gd_path = gradient_descent(x0, eta, iterations)
sgd_path = stochastic_gradient_descent(x0, eta, iterations, x_data)

# Visualization
plt.figure(figsize=(14, 10))
plt.plot(x_data, y_data, label='Cost Function')
plt.plot(gd_path, cost(np.array(gd_path)), marker='o', label='GD Path', color='red')
plt.plot(sgd_path, cost(np.array(sgd_path)), marker='x', label='SGD Path', color='green')
plt.xlabel('x')
plt.ylabel('Cost f(x)')
plt.title('GD vs. SGD Optimization Path')
plt.legend()
plt.grid(True)
plt.show()