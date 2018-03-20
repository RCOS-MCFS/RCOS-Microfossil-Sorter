from approaches.approach import Approach

import numpy as np
import logging
import progressbar

class Multiclass_Logistic_Regression(Approach):
    def classify_datum(self, W, row):
        val = [np.dot(W[k], row) for k in range(np.shape(W)[0])]
        classification_vector = [1 if k == val.index(max(val)) else 0 for k in range(self.K)]
        return classification_vector

    def train(self, training_data, learning_rate=0.05, max_epochs=10):
        '''
        :param training_data: A matrix containing the data to be examined, with the final column being the label of this data
        :param learning_rate: The rate at which the intro changes.
        :param max_epochs: Maximum number of training iterations
        :return: Kxd matrix of weights
        '''

        # Way of escaping overflow errors.
        def my_exp(z):
            return np.maximum(z, 0) + np.log(np.exp(-np.absolute(z)) + 1)

        # Gradient used in the multiclass logistic regression equation.
        def gradient(Theta, X, Y, k, K):
            M = np.shape(X)[0]
            grad_val = -1*sum([X[m]*(Y[m][k] - ((my_exp(np.dot(Theta[k], X[m])))/sum([my_exp(np.dot(Theta[k_prime], X[m]))
                        for k_prime in range(K)]))) for m in range(M)])
            return grad_val

        classifications = training_data[:, -1]
        training_data = training_data[:, 0:-1]
        m = len(training_data)
        d = np.shape(training_data[0])[0]

        K = int(max(classifications))
        labels = [self.label_to_vector(c, K) for c in classifications]

        self.K = K

        # Initialize our progress bar.
        progressbar.streams.wrap_stderr()
        logging.basicConfig()
        bar = progressbar.ProgressBar()

        # Initialize Theta to random small values.
        Theta = np.random.random_sample((K, d))/1000
        for i in bar(range(max_epochs)):
            W_new = np.copy(Theta)
            for k in range(K):
                W_new[k] = Theta[k] - (learning_rate * gradient(Theta, training_data, labels, k, K))
            Theta = np.copy(W_new)

        self.weights = Theta

    def label_to_vector(self, x, K):
        '''
        :param x: Integer to be transoformed into a vector
        :param K: Number of possible classifications
        :return: A matrix of 0 and 1 of dimension K representing x.
        '''
        label = [0 for i in range(K)]
        label[int(x)-1] = 1
        return label