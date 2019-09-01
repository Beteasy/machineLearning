import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = np.genfromtxt("data.csv", delimiter=",")
# x_data， y_data原来是一维的，给他增加一个维度，将其变为二维的
# 原因是后面的fit要求传进去的数据是二维的
x_data = data[:, 0, np.newaxis]
y_data = data[:, 1, np.newaxis]

# create and fit model
model = LinearRegression()
model.fit(x_data, y_data)

plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, model.predict(x_data), 'r')
plt.show()