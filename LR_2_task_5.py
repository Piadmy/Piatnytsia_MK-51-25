import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Генерація даних
np.random.seed(0)
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.7 * X**2 + X + 3 + np.random.randn(m, 1)

# Лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin_pred = lin_reg.predict(X)

# Поліноміальна регресія
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)
y_poly_pred = lin_reg_poly.predict(X_poly)

# Коефіцієнти
print("Лінійна модель:")
print("y =", lin_reg.coef_[0][0], "* X +", lin_reg.intercept_[0])

print("\nПоліноміальна модель:")
print("y =", lin_reg_poly.coef_[0][1], "* X^2 +",
      lin_reg_poly.coef_[0][0], "* X +",
      lin_reg_poly.intercept_[0])

# Метрики
print("\nLinear R2:", r2_score(y, y_lin_pred))
print("Polynomial R2:", r2_score(y, y_poly_pred))

# Графік
plt.scatter(X, y, color='blue', label='Дані')

X_sorted = np.sort(X, axis=0)
y_lin_sorted = lin_reg.predict(X_sorted)

X_poly_sorted = poly_features.transform(X_sorted)
y_poly_sorted = lin_reg_poly.predict(X_poly_sorted)

plt.plot(X_sorted, y_lin_sorted, color='red', label='Лінійна регресія')
plt.plot(X_sorted, y_poly_sorted, color='green', label='Поліноміальна регресія')

plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Порівняння регресій")
plt.show()
