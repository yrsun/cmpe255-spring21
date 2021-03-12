# Assignment 1

In this assigment, you will be building models to predict house prices using the [Boston housing dataset](https://www.kaggle.com/vikrishnan/boston-house-prices).

# Output:

Q1 LSTAT:
RMSE = 6.1724146152405135
R2 = 0.4886979007906852
Q2 2 LSTAT:
RMSE = 5.61957012958762
R2 = 0.5761876727750421
Q2 20 LSTAT:
RMSE = 26.999119619888152
R2 = -8.782877153252805
Q3 LSTAT RM PTRATIO:
RMSE = 5.100216850500207
R2 = 0.650904156861472
Adjusted_R2 = 0.6438278897708262

## I - Linear Regression

* Build a model using a linear regression (Scikit-learn) algorithm to predict house prices. You can pick a feature from the dataset to work with the model.

```python
from sklearn.linear_model import LinearRegression
```

> Y = C + w * X

* Plot the data with the best fit line.
* Calculate a RMSE score.
* Calculate a R-squared score.


## II - Polynomial Regression

* Build a model using a Polynomial regression algorithm to predict house prices. Keep the same feature you selected from the previous part to work with the polynomial model. 

> Y = C + w<sub>1</sub> * X + w<sub>2</sub> * X<sup>2</sup> 

```python
from sklearn.preprocessing import PolynomialFeatures
```

* X<sup>2</sup> is only a feature, but the curve that we are fitting is in quadratic form.
* Plot the best 2nd degree polynomail curve.
* Calculate a RMSE score.
* Calculate a R-squared score.
* Plot another diagram for degree=20.


## III - Multiple Regression

* Build a model using a multiple regression algorithm to predict house prices. Select 3 or more features to work with the model. 

> Y = C + w<sub>1</sub> * X<sub>1</sub> + w<sub>2</sub> * X<sub>2</sub> + w<sub>3</sub> * X<sub>3</sub>

* Plot the curve.
* Calculate a RMSE score.
* Calculate a R-squared score.
* Calculate an adjusted R-squared score.
