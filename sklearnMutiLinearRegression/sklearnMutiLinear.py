import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

data = np.genfromtxt("Delivery.csv", delimiter=",")

x_data = data[:, :-1]
y_data = data[:, -1]

model = LinearRegression()
model.fit(x_data, y_data)

ax = plt.figure().add_subplot(111, projection="3d")
ax.scatter(x_data[:,0], x_data[:,1], y_data, c='r', marker='o', s=100)
x0 = x_data[:, 0]
x1 = x_data[:, 1]
# 生成网络矩阵
x0, x1 = np.meshgrid(x0, x1)
z = model.intercept_ + (x0*model.coef_[0] + x1*model.coef_[1])
ax.plot_surface(x0, x1, z)
ax.set_xlabel("M")
ax.set_ylabel("Num")
ax.set_zlabel("T")
plt.show()