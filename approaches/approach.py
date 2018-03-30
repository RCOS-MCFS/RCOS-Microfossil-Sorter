from sklearn import svm

import numpy as np
import logging
import os
import pickle
import progressbar


class Approach():
    def __init__(self):
        self.weights = None

    def assess_accuracy(self, test_data):
        '''
        :param test_data: Numpy data to be used as testing, with the final value in each row being the label
        :return: A value between 0.00 and 1.00 representing the accuracy of the model.
        '''
        labels = test_data[:,-1]
        test_data = test_data[:,0:-1]
        classifications = self.classify(test_data)
        correct_count = sum([int(classifications[i] == labels[i]) for i in range(len(labels))])
        return correct_count/len(classifications)

    def classify(self, data):
        '''
        :param data: A list of numpy matrices to be classified, or single matrix
        :return: A list of the generated classifications for each numpy matrix passed
        '''
        if len(np.shape(data)) == 2:
            return [self.classify_datum(self.weights, d) for d in data]
        else:
            return self.classify_datum(self.weights, data)

    def load_weights(self, input):
        '''
        :param input: The pickle file to be used for loading these weights
        :return: None
        '''
        if not os.path.isfile(input):
            raise ValueError("Input path " + input + "does not link to an actual file.")
        if ".p" not in input:
            raise ValueError("Input file must terminate in .p, indicating a pickle file")
        self.weights = pickle.load(input, "rb")

    def get_weights(self):
        '''
        :return: The weights for this model
        '''
        if self.weights:
            return self.weights
        else:
            raise NameError("No weights currently exist.")

    def set_weights(self, new_weights):
        '''
        :param new_weights: Weights being passed in.
        :return: None
        '''
        if new_weights and len(new_weights) > 0:
            self.weights = new_weights
        else:
            raise ValueError("No data passed")

    def write_weights(self, output):
        '''
        :param output: The filename to which these weights are being printed.
        :return: None
        '''
        if self.weights == None:
            raise NameError("No weights currently exist.")
        if ".p" not in output:
            raise ValueError("Output must end in the extension .p")
        pickle.dump(self.weights, open(output, "wb"))

class Multiclass_Logistic_Regression(Approach):
    def classify_datum(self, W, row):
        val = [np.dot(W[k], row) for k in range(np.shape(W)[0])]
        classification_vector = [1 if k == val.index(max(val)) else 0 for k in range(self.K)]
        return classification_vector.index(max(classification_vector))

    def train(self, training_data, learning_rate=0.05, max_epochs=1000):
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


class Perceptron(Approach):

    def classify_datum(self, weights, row, act_type='linear'):
        def linear(value):
            return 1.00 if value > 0.00 else 0.00
        # A row by any other name ...
        activation_val = sum([weights[i]*val for i, val in enumerate(row)])
        return linear(activation_val)

    def train(self, training_data, learning_rate=0.005, epochs=1000, act_type='linear'):
        '''
        :param training_images: A numpy matrix of the average colors for images in the training set, with the final column
        representing their type, 1 being bone and 0 being rock.
        :return: A list containing the trained weights for the perceptron.
        '''
        # Break classification data from training data
        classifications = training_data[:, -1]
        training_data = training_data[:,0:-1]

        # Establish original weights.
        n, d = np.shape(training_data)
        weights = [0.001 for i in range(d+1)] # +1 for the bias


        # Initialize our progress bar.
        progressbar.streams.wrap_stderr()
        logging.basicConfig()
        bar = progressbar.ProgressBar()
        for epoch in bar(range(epochs)):
            learning_rate *= .995
            for i, row in enumerate(training_data):
                prediction = self.classify_datum(weights, row, act_type)
                error = classifications[i] - prediction
                for j, val in enumerate(row):
                    weights[j] += learning_rate*error*val
                weights[-1] += learning_rate*error
        self.weights = weights

class Sklearn_SVM(Approach):

    def classify_datum(self, clf, row):
        retval = clf.predict([row])[0]
        return retval

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