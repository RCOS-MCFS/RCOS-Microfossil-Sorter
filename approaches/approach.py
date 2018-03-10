class Approach():
    def __init__(self):
        self.weights = None

    def assess_accuracy(self, test_data):
        labels = test_data[:,-1]
        test_data = test_data[:,0:-1]
        classifications = self.classify(test_data)
        correct_count = sum([int(classifications[i] == labels[i]) for i in range(len(labels))])
        return correct_count/len(classifications)

    def classify(self, data):
        if len(data) == 1:
            return self.classify_datum(self.weights, data)
        if len(data) > 1:
            return [self.classify_datum(self.weights, d) for d in data]
        else:
            raise ValueError("No data passed.")

    def get_weights(self):
        if self.weights:
            return self.weights
        else:
            raise NameError("Weights have not yet been trained.")