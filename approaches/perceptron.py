# RCOS - MCFS
# For color-based discrimination, a perceptron may be more than enough.
from approaches.approach import Approach

import logging
import numpy as np
import progressbar

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
            if epoch%30 == 0:
                print(learning_rate)
            for i, row in enumerate(training_data):
                prediction = self.classify_datum(weights, row, act_type)
                error = classifications[i] - prediction
                for j, val in enumerate(row):
                    weights[j] += learning_rate*error*val
                weights[-1] += learning_rate*error
        self.weights = weights