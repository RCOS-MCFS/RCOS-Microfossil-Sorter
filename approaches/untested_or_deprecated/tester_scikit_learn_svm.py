# RCOS MCFS
# Testing use of scikit-learn's svm function

import numpy as np
from sklearn import svm
#from sklearn.svm import SVC
import csv
import sys
import random


raw_data = [] # 1st bone data, 2nd rock data
train_data = []
train_values = []
test_data = []
test_values = []
test_results = [] # Output

# Read in training data
def load_csv(filename):
    print(filename + "\n")
    #print(os.path.join(path, filename) + "\n")
    with open(filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            raw_data.append(row)

print(sys.argv[1] + "\n")
load_csv(sys.argv[1])

new_data = []
for x in raw_data:
    if x:
        new_data.append(x)
raw_data = new_data

# Split into training and testing data, 50:50 ratio
data_size = len(raw_data)
train_array = random.sample(range(0, data_size), int(data_size/2))
for i in train_array:
    train_data.append(raw_data[i][0:3])
    train_values.append(raw_data[i][3])

for i in range(0, data_size):
    if i not in train_array:
        test_data.append(raw_data[i][0:3])
        test_values.append(raw_data[i][3])

print("TRAIN DATA:\n")
print(train_data)
print(train_values)
print("TEST DATA:\n")
print(test_data)
print(test_values)

# train and test svm


clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

clf.fit(train_data, train_values)

result = clf.predict(test_data)

# Print results

print("PREDICTION:\n")
print(len(result))
print("\n")
print(result)
print("ACTUAL:\n")
print(test_values)
print("\n")
print(len(test_values))
print("\n")


correct = 0
for a in range(0, len(test_values)):
    if test_values[a] == result[a]:
        correct += 1


print("PERCENT:\n")
print(correct / len(test_values))