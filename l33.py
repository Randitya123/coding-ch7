import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

from sklearn. linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error

data=pd.read_csv("C:/Users/KIKE/OneDrive/Desktop/New folder/data.csv")

# Select two features analogous to LSTAT, RM in Boston example
x = data[["petal_width","sepal_width"]]
Y = data["petal_length"]

# Train / Test split
X_train, X_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.2, random_state=5)


# --- Multivariable Linear Regression -..
lin_model = LinearRegression()
lin_model. fit(X_train, Y_train)
y_pred_lin = lin_model.predict(X_test)
rmse_lin = np.sqrt(mean_squared_error(Y_test, y_pred_lin))
print(f"RMSE (Linear Regression): {rmse_lin: .4f}")


# --- Multivariable Linear Regression ---
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)
y_pred_lin = lin_model. predict(X_test)
rmse_lin = np.sqrt(mean_squared_error(Y_test, y_pred_lin))
print(f"RMSE (Linear Regression): {rmse_lin: 4F}")

# --- Polynomial Regression (degree = 2) ---
poly = PolynomialFeatures (degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test                                                                   )
poly_model= LinearRegression()
poly_model.fit(X_train_poly,Y_train)
y_pred_poly=poly_model.predict(X_test_poly)
rmse_poly=np.sqrt(mean_squared_error(Y_test,y_pred_poly))
print(f"RMSE (Polynomial Regression): {rmse_poly:.4f}")

