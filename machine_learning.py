import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


### x, y are known data input. x_predict is value want to predict
def linear_reg(x,y, x_predict):
    X = np.stack([x['pace']], axis=1)  # join 1D array to 2D
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    model = LinearRegression(fit_intercept=True)  # Create a model
    model.fit(X_train, y_train)     #Train the data
    print("lineary regression score on train: ",model.score(X_train, y_train))
    print("linear regression score: ",model.score(X_valid, y_valid))
    ##Predict data for x_predict value
    # X_fit =  np.stack([x_predict['step'], x_predict['height']], axis=1)
    X_fit = x_predict
    y_fit = model.predict(X_fit)
    print(y_fit)

    plt.plot (x, y, 'b.')
    plt.plot(X_fit, y_fit, 'g.')
    plt.plot(x, model.predict(X), 'r-')
    plt.show()
    return

def polynomial_reg(x,y, x_predict):
    model = make_pipeline(
        PolynomialFeatures(degree=5, include_bias=True),
        LinearRegression(fit_intercept=False)
    )
    X = np.stack([x['pace']], axis=1)  # join 1D array to 2D
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    model.fit(X_train, y_train)
    print("poly regression score on train: ",model.score(X_train, y_train))
    print("poly regression score: ",model.score(X_valid, y_valid))   

    # model.fit(X, y)
    #Predict data for x_predict value
    X_fit = x_predict
    y_fit = model.predict(X_fit)
    print(y_fit)
    return


def classification_model(x, y, x_predict):
    X_train, X_valid, y_train, y_valid = train_test_split(x, y)

    bayes_model = GaussianNB()
    bayes_model.fit(X_train, y_train)
    y_bayes_fit = bayes_model.predict(x_predict)

    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    y_knn_fit = bayes_model.predict(x_predict)


    rf_model = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10)
    rf_model.fit(X_train, y_train)
    y_rf_fit = bayes_model.predict(x_predict)


    ## If the data is imbalance between male and female. We can rebalance it
    # female = data[data['gender'] == 1]
    # male = data[data['gender'] == 0].sample(n=male.shape[0])
    # balanced_data = female.append(male)
    
    print('Bayes Model: ',bayes_model.score(X_valid, y_valid))
    print('KNN Model: ',knn_model.score(X_valid, y_valid))
    print('Random Forest Model: ',rf_model.score(X_valid, y_valid))

    print("Bayes predict: ",y_bayes_fit)
    print("KNN predict: ",y_knn_fit)
    print("Random Forest predict: ",y_rf_fit)

    return



if __name__ == "__main__":
    data = pd.read_csv('mldata.csv')
    # x = data[['pace']]
    # y = data['height']
    # x_predict= np.array([[0.9]])
    # linear_reg(x, y , x_predict)
    # polynomial_reg(x, y , x_predict)

    x = data[['step', 'pace']]
    y = data['range']
    x_predict =[[15, 0.9]]
    classification_model(x,y, x_predict)