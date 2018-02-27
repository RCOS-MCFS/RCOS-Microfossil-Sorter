from collections import Counter
import numpy as np

def classify(W, data_vector):
    assert(np.shape(W)[1] == np.shape(data_vector)[0])
    vals = [np.dot(W[k], data_vector) for k in range(np.shape(W)[0])]
    classification_vector = [1 if k == val.index(max(vals)) else 0 for k in range(K)]
    return classification_vector

def assess_accuracy(W, testing_data):
    # Split data appropriately and generate labels in expected format, as well as variables needed.
    classifications_numerical = testing_data[:, -1]
    testing_data = testing_data[:, 0:-1]
    m = len(testing_data)
    d = np.shape(testing_data[0])[0]

    K = max(classifications_numerical)
    classifications_vector = [label_to_vector(c, K) for c in classifications_numerical]

    correct = 0
    correct_count = Counter()
    total_count = Counter(classifications_numerical)
    for i, img in enumerate(testing_data):
        if classify(W, img) == classifications_vector[i]:
            correct_count[classifications_numerical[i]] += 1
            correct += 1

    print("ERROR RATES:")
    for i in range(K):
        v = i+1
        print(str(v) + ': ' + str(correct_count[v]/total_count[v]))
    print("Overall accuracy: " + str(correct/len(classifications_numerical)))

def create_weights(training_data, learning_rate=0.05, max_epochs=10):
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

    K = max(classifications)
    labels = [label_to_vector(c, K) for c in classifications]

    # Initialize Theta to random small values.
    Theta = np.random.random_sample((K, d))/1000
    for i in range(max_epochs):
        W_new = np.copy(Theta)
        for k in range(K):
            W_new[k] = Theta[k] - (learning_rate * gradient(Theta, training_data, labels, k, K))
        print(np.linalg.norm(W_new - Theta))
        Theta = np.copy(W_new)

    return Theta

def label_to_vector(x, K):
    '''
    :param x: Integer to be transoformed into a vector
    :param K: Number of possible classifications
    :return: A matrix of 0 and 1 of dimension K representing x.
    '''
    label = [0 for i in range(K)]
    label[int(x)-1] = 1
    return label