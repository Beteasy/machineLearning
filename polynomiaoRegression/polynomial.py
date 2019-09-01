import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = np.genfromtxt("job.csv", delimiter=",")
x_data = data[1:, 1, np.newaxis]
y_data = data[1:, 2, np.newaxis]
# plt.scatter(x_data, y_data)
# plt.show()

poly_reg  =PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x_data)
lin_reg = LinearRegression()
lin_reg.fit(x_poly, y_data)

plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, lin_reg.predict(poly_reg.fit_transform(x_data)), c='r')
plt.show()