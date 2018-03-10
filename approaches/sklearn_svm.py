# SVM using the built in functions in Skikit
from approaches.approach import Approach

from sklearn import svm
import csv
import sys
import random

class Sklearn_SVM(Approach):

    def classify_datum(self, clf, row):
        return clf.predict(row)[0]

    def train(self, training_data):
        train_values = training_data[:, -1]
        train_data = training_data[:, 0:-1]

        clf = svm.SVC(C=1.0, cache_size=2000, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=5, gamma='auto', kernel='poly',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)

        clf.fit(train_data, train_values)

        # I realize that weights may not be the most appropriate terms for this model, but it was the best
        # one for the superclass which would then cover all of the lower cases.
        self.weights = clf
