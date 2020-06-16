#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree


class DecisionTree:
    def __init__(self, max_depth=9, min_size=4):
        self.max_depth = max_depth
        self.min_size = min_size

    def fit(self, X, y):
        dataset = np.column_stack([X, y])
        self.tree = self.build_tree(dataset)

    def predict(self, X):
        if len(X.shape) == 1:
            return self.predict_one(self.tree, X)
        else:
            y_pred = []
            for x in X:
                y_pred.append(self.predict_one(self.tree, x))
            return y_pred

    def print_tree(self, node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % (depth * ' ', (node['index'] + 1), node['value']))
            self.print_tree(node['left'], depth + 1)
            self.print_tree(node['right'], depth + 1)
        else:
            print('%s[%s]' % (depth * ' ', node))

    def gini_index(self, groups, classes):
        n_instanse = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            gini += (1.0 - score) * (size / n_instanse)
        return gini

    def test_splitting(self, index, value, dataset):
        left, rigth = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                rigth.append(row)
        return left, rigth

    def get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None

        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self.test_splitting(index, row[index], dataset)
                gini_id = self.gini_index(groups, class_values)
                if gini_id < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini_id, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    def to_term(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    def splitting(self, node, depth):
        max_depth, min_size, = self.max_depth, self.min_size

        left, right = node['groups']
        del (node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.to_term(left + right)
            return
        if depth >= max_depth:
            node['left'], node['right'] = self.to_term(left), self.to_term(right)
            return

        if len(left) <= min_size:
            node['left'] = self.to_term(left)
        else:
            node['left'] = self.get_split(left)
            self.splitting(node['left'], depth + 1)

        if len(right) <= min_size:
            node['right'] = self.to_term(right)
        else:
            node['right'] = self.get_split(right)
            self.splitting(node['right'], depth + 1)

    def build_tree(self, train):
        root = self.get_split(train)
        self.splitting(root, 1)
        return root

    def predict_one(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict_one(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_one(node['right'], row)
            else:
                return node['right']


if __name__ == '__main__':
    mobile_data = pd.read_csv('newRain.csv')
    mobile_data.drop(['Unnamed: 0'], axis='columns', inplace=True)
    
    X, Y = mobile_data.drop(['RainTomorrow'], axis=1), mobile_data['RainTomorrow']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    model = DecisionTree()
    model.fit(x_train.values, y_train.values)
    y_pred = model.predict(x_train.values)
    print(f'Train accuracy score: {accuracy_score(y_train.values, y_pred)}')
    y_pred = model.predict(x_test.values)

    print(f'Test accuracy score: {accuracy_score(y_test.values, y_pred)}')

    dt = tree.DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    print(f'Sklearn accuracy score: {accuracy_score(y_test, y_pred)}')

