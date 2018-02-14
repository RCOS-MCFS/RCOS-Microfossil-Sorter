# RCOS - MCFS
# For color-based discrimination, a perceptron may be more than enough.
import numpy as np
import random

def predict(weights, row, act_type):
    def linear(value):
        return 1.00 if value > 0 else 0.00

    activation_val = sum([weights[i]*val for i, val in enumerate(row)]) + weights[-1]
    return linear(activation_val)


def create_perceptron(training_data, learning_rate=0.01, epochs=100, act_type='linear'):
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
    weights = [0.00 for i in range(d+1)] # +1 for the bias
    for epoch in range(epochs):
        for i, row in enumerate(training_data):
            prediction = predict(weights, row, act_type)
            error = classifications[i] - prediction
            for i, val in enumerate(row):
                weights[i] += learning_rate*error*val
            weights[-1] += learning_rate*error
    return weights

def test_accuracy(weights, test_data, act_type='linear'):
    classifications = test_data[:,-1]
    test_data = test_data[:,0:-1]

    accurate_count = sum([int(classifications[i] == predict(weights, test_data[i], act_type))
                          for i in range(len(test_data))])
    return accurate_count/len(test_data)