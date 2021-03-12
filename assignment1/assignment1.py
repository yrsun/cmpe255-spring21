import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import operator

boston = load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data['PRICE'] = boston.target


# Question 1
X = data['LSTAT'].values.reshape(-1, 1)
Y = data['PRICE']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

reg1 = linear_model.LinearRegression()
reg1.fit(X_train, Y_train)
Y_pred1 = reg1.predict(X_test)

mae = metrics.mean_absolute_error(Y_test, Y_pred1)
mse = metrics.mean_squared_error(Y_test, Y_pred1)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(Y_test, Y_pred1)
print('Q1 LSTAT:')
print(f'RMSE = {rmse}')
print(f'R2 = {r2}')

plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color='black')
plt.plot(X_test, Y_pred1)
plt.xlabel('LSTAT')
plt.ylabel('PRICE')
plt.show()

# Question 2
poly = PolynomialFeatures(2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.fit_transform(X_test)
reg2 = linear_model.LinearRegression()
reg2.fit(X_poly_train, Y_train)
Y_pred2 = reg2.predict(X_poly_test)

mae = metrics.mean_absolute_error(Y_test, Y_pred2)
mse = metrics.mean_squared_error(Y_test, Y_pred2)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(Y_test, Y_pred2)
print('Q2 2 LSTAT:')
print(f'RMSE = {rmse}')
print(f'R2 = {r2}')

plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color='black')
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X_test, Y_pred2), key = sort_axis)
X_test2, Y_pred2 = zip(*sorted_zip)
plt.plot(X_test2, Y_pred2)
plt.xlabel('LSTAT')
plt.ylabel('PRICE')
plt.show()

poly = PolynomialFeatures(20)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.fit_transform(X_test)
reg2 = linear_model.LinearRegression()
reg2.fit(X_poly_train, Y_train)
Y_pred2 = reg2.predict(X_poly_test)

mae = metrics.mean_absolute_error(Y_test, Y_pred2)
mse = metrics.mean_squared_error(Y_test, Y_pred2)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(Y_test, Y_pred2)
print('Q2 20 LSTAT:')
print(f'RMSE = {rmse}')
print(f'R2 = {r2}')

plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color='black')
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X_test, Y_pred2), key = sort_axis)
X_test2, Y_pred2 = zip(*sorted_zip)
plt.plot(X_test2, Y_pred2)
plt.xlabel('LSTAT')
plt.ylabel('PRICE')
plt.show()

# Question 3
X = data[['LSTAT', 'RM', 'PTRATIO']]
Y = data['PRICE']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

reg3 = linear_model.LinearRegression()
reg3.fit(X_train, Y_train)
Y_pred3 = reg3.predict(X_test)

mae = metrics.mean_absolute_error(Y_test, Y_pred3)
mse = metrics.mean_squared_error(Y_test, Y_pred3)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(Y_test, Y_pred3)
adj_r2 = 1 - (1-r2) * ((len(X_test) - 1) / (len(X_test) - len(X_test.values[0]) - 1))
print('Q3 LSTAT RM PTRATIO:')
print(f'RMSE = {rmse}')
print(f'R2 = {r2}')
print(f'Adjusted_R2 = {adj_r2}')

