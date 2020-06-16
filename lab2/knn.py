#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import copy
import math
import pandas as pd
from random import randrange
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    def __init__(self, n_neghbors=5, n_folds=5):
        self.n_neigbors = n_neghbors
        self.n_folds = n_folds

    def euclidean_distance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2
        return math.sqrt(distance)

    # Locate the most similar neighbors
    def get_neighbors(self, train, test_row, num_neighbors):
        distances = []
        for train_row in train:
            dist = self.euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = []
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    # Make a prediction with neighbors
    def predict_classification(self, train, test_row):
        neighbors = self.get_neighbors(train, test_row, self.n_neigbors)
        output = [each[-1] for each in neighbors]
        prediction = max(set(output), key=output.count)
        return prediction

    # Split a dataset into k folds
    def cross_val_split(self, dataset, n_folds):
        dataset_split = []
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for _ in range(n_folds):
            fold = []
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    # Calculate accuracy
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual))

    # Cross validation split
    def evaluate_algorithm(self, dataset, algo, n_folds, *args):
        folds = self.cross_val_split(dataset, n_folds)
        scores = []
        for i in range(len(folds)):
            fold = folds[i]
            train_set = list(folds[:i])
            if len(folds[i + 1:]) != 0:
                train_set.extend(folds[i + 1:])
            train_set = sum(train_set, [])
            test_set = []
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = algo(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores

    def k_nearest_neighbors(self, train, test):
        predictions = []
        for row in test:
            output = self.predict_classification(train, row)
            predictions.append(output)
        return predictions

    def fit(self, x, y):
        # x = pd.concat([x, y], axis=1)
        return self.evaluate_algorithm(x, self.k_nearest_neighbors, self.n_folds)

    def predict(self, x_train, x_test):
        labels = []
        for row in x_test:
            label = self.predict_classification(x_train, row)
            labels.append(label)
        return labels


if __name__ == '__main__':
    mobile_data = pd.read_csv('newRain.csv')
    mobile_data.drop(['Unnamed: 0'], axis='columns', inplace=True)
    
    X, Y = mobile_data.drop(['RainTomorrow'], axis=1), mobile_data['RainTomorrow']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    x_1_train = copy.deepcopy(x_train)
    x_train = pd.concat([x_train, y_train], axis=1)

    num_neighbors = 5
    n_folds = 5
    model = KNN(num_neighbors, n_folds)
    # scores = model.evaluate_algorithm(x_train.values, model.k_nearest_neighbors, n_folds, num_neighbors)
    scores2 = model.fit(x_train, y_train)
    y_pred = model.predict(x_train, x_test)
    print(f'CV scores: {scores2}')
    mn = sum(scores2) / float(len(scores2))
    print(f'Train data accuracy: {mn}')

    test_score = model.accuracy_metric(y_test.values, y_pred)
    print(f'Test data accuracy: {test_score}')

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(x_1_train, y_train)
    y_pred = neigh.predict(x_test)
    print(f'Sklearn accuracy score: {accuracy_score(y_test, y_pred)}')

