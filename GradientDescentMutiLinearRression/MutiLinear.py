import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.genfromtxt("Delivery.csv", delimiter=",")
x_data = data[:, :-1]
y_data = data[:, -1]

lr = 0.0001
bias = 0
weight0 = 0
weight1 = 0
epochs = 1000

def gradientDescent(x_data, y_data, bias, weight0, weight1, epochs, lr):
   # the total number data
    num = float(len(x_data))
    for i in range(epochs):
        bias_grad = 0
        weight0_grad = 0
        weight1_grad = 0
        # 计算梯度的总平均
        for j in range(0, len(x_data)):
            bias_grad += -(1/num) * (y_data[j] - (weight0 * x_data[j, 0] + weight1 * x_data[j, 1] + bias))
            weight0_grad += -(1/num) * (y_data[j] - (weight0 * x_data[j, 0] + weight1 * x_data[j, 1] + bias)) * x_data[j, 0]
            weight1_grad += -(1/num) * (y_data[j] - (weight0 * x_data[j, 0] + weight1 * x_data[j, 1] + bias)) * x_data[j, 1]

        bias = bias - (lr * bias_grad)
        weight0 = weight0 - (lr * weight0_grad)
        weight1 = weight1 - (lr * weight1_grad)
    return bias, weight0, weight1

bias, weight0, weight1 = gradientDescent(x_data, y_data, bias, weight0, weight1, epochs, lr)
print(bias, weight0, weight1)

ax = plt.figure().add_subplot(111, projection="3d")
ax.scatter(x_data[:,0], x_data[:,1], y_data, c='r', marker='o', s=100)
x0 = x_data[:, 0]
x1 = x_data[:, 1]
# 生成网络矩阵
x0, x1 = np.meshgrid(x0, x1)
z = weight0*x0 + weight1*x1 + bias
ax.plot_surface(x0, x1, z)
ax.set_xlabel("M")
ax.set_ylabel("Num")
ax.set_zlabel("T")
plt.show()
# plt.plot(x_data, y_data, 'b.')
# plt.plot(x_data, )