import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class LogRegression:
    def __init__(self, C=0):
        self.theta = []
        self.costs = []
        self.lambd = C

    def sigmoid(self, x):
        e = 1.0 + math.exp(1) ** (-1.0 * x)
        f = 1.0 / e
        return f

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        if e.ndim == 1:
            return e / np.sum(e, axis=0)
        else:
            return e / np.array([np.sum(e, axis=1)]).T

    def fit(self, x, y, max_iter=100, alpha=0.01):
#         x = x.values
        y = pd.get_dummies(y)
#         y = y.values
        x = np.insert(x, 0, 1, axis=1)
        costs = np.zeros(max_iter)

        theta = np.zeros((x.shape[1], y.shape[1]))
        for epoch in range(max_iter):
            costs[epoch], grad = self.cost(theta, x, y)
            theta = theta - alpha * grad

        self.theta = theta
        self.costs = costs

    def cost(self, theta, x, y):
        multi = np.dot(x, theta)
        h = np.array(list(map(lambda i: list(self.softmax(i)), multi)))
        m = len(y)
        cost = 1 / m * np.sum([(-y_i * np.log(h_i)) for y_i, h_i in zip(y, h)]) \
               + np.sum(theta ** 2) * self.lambd / (2 * m)  # With regularization
        grad = np.dot(x.T, h - y) / m + theta * self.lambd / m
        return cost, grad

    def predict(self, x):
        x = np.insert(x, 0, 1, axis=1)
        pred = np.dot(x, self.theta)
        pred = np.array(list(map(lambda i: list(self.softmax(i)), pred)))
        pred = np.argmax(pred, 1)
        return pred

    def score(self, y_test, y_pred):
        return accuracy_score(y_test, y_pred)
    
    def __str__(self):
        return self.pressid


if __name__ == '__main__':
    mobile_data = pd.read_csv('newDiamonds.csv')df
    
    X, Y = mobile_data.drop(['price'], axis=1), mobile_data['price']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    model = LogRegression()
#     x_train = x_train.values
#     y_train = pd.get_dummies(y_train)
#     y_train = y_train.values
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test.values)
    print(f"Test Accuracy: {model.score(y_test, y_pred)}")

    y_pred = model.predict(x_train.values)
    print(f"Train Accuracy: {model.score(y_train, y_pred)}")

    logreg = LogisticRegression(C=1e5)
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    print(f'Sklearn accuracy score: {accuracy_score(y_test, y_pred)}')