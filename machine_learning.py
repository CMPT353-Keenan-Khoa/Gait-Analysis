import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split



### x, y are known data input. x_predict is value want to predict
def linear_reg(x,y, x_predict):
    X = np.stack([x], axis=1)  # join 1D array to 2D
    model = LinearRegression(fit_intercept=True)  # Create a model
    model.fit(X, y)     #Train the data

    ##Predict data for x_predict value
    X_fit =  np.stack([x_predict], axis=1)
    y_fit = model.predict(X_fit)
    print(y_fit)
    return

def polynomial_reg(x,y, x_predict):
    model = make_pipeline(
        PolynomialFeatures(degree=n, include_bias=True),
        LinearRegression(fit_intercept=False)
    )
    X = np.stack([x], axis=1)
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    # model.fit(X_train, y_train)
    # print(model.score(X_train, y_train))
    # print(model.score(X_valid, y_valid))   

    model.fit(X, y)
    ##Predict data for x_predict value
    X_fit =  np.stack([x_predict], axis=1)
    y_fit = model.predict(X_fit)
    print(y_fit)
    return




if __name__ == "__main__":
    pass
    