#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from decTree import DecisionTree
from collections import  Counter
import math
class RandomForest:
    def __init__(self,n_estimators=10,samples = None, max_depth=9,min_size=4):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_size = min_size
        self.samples = samples
    def fit(self,X,y):
        self.decision_trees = []

        for _ in range(self.n_estimators):

            ind_row = np.random.choice(X.shape[0],X.shape[0]) if self.samples == None else np.random.choice(X.shape[0],self.samles)
            ind_col = np.random.choice(X.shape[1],int(math.sqrt(X.shape[1])),replace = False)
            X_est = X[ind_row]
            X_est = X_est[:,ind_col]

            y_est = y[ind_row]
            dt = DecisionTree(max_depth=self.max_depth, min_size=self.min_size)
            dt.fit(X_est,y_est)
            self.decision_trees.append(dt)
    def predict(self,X):
        result = None
        if len(X.shape) == 1:
            result = self.__predict_one(X)
        else:
            y_preds = []
            for x in X:
                y_pred = self.__predict_one(x)
                y_preds.append(y_pred)
            result = y_preds
        return  result

    def __predict_one(self,x):
        y_preds = []
        for dt in self.decision_trees:
            y_pred = dt.predict(x)
            y_preds.append(y_pred)
        voices = Counter(y_preds)
        y_result = voices.most_common()[0][0]
        return  y_result

if __name__ == '__main__':
    mobile_data = pd.read_csv('newRain.csv')
    mobile_data.drop(['Unnamed: 0'], axis='columns', inplace=True)
    
    X, Y = mobile_data.drop(['RainTomorrow'], axis=1), mobile_data['RainTomorrow']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    model = RandomForest()
    model.fit(x_train.values, y_train.values)
    y_pred = model.predict(x_train.values)
    print(f'Train accuracy score: {accuracy_score(y_train.values, y_pred)}')
    y_pred = model.predict(x_test.values)

    print(f'Test accuracy score: {accuracy_score(y_test.values, y_pred)}')

