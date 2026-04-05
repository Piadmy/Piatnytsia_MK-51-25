import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Дані
np.random.seed(0)
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.7 * X**2 + X + 3 + np.random.randn(m, 1)

# Розбиття
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Функція побудови кривих навчання
def plot_learning_curves(model, X_train, y_train, X_val, y_val):
    train_errors, val_errors = [], []
    
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    
    plt.plot(np.sqrt(train_errors), "r-", label="Train")
    plt.plot(np.sqrt(val_errors), "b-", label="Validation")
    
    plt.xlabel("Кількість навчальних прикладів")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("Криві навчання для поліноміальної моделі 2-го ступеня")
    plt.show()

# Поліноміальна модель 2-го ступеня
polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression())
])

plot_learning_curves(polynomial_regression, X_train, y_train, X_val, y_val)
