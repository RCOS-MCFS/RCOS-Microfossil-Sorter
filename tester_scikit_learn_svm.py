# RCOS MCFS
# Testing use of scikit-learn's svm function

import numpy as np
from sklearn import svm
#from sklearn.svm import SVC
import csv
import sys
import random


rawdata = [] # 1st bone data, 2nd rock data
traindata = []
trainvalues = []
testdata = []
testvalues = []
testresults = [] # Output


# Read in training data
def load_csv(filename):
    print(filename + "\n")
    #print(os.path.join(path, filename) + "\n")
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            rawdata.append(row)



print(sys.argv[1] + "\n")
load_csv(sys.argv[1])

new_data = []
for x in rawdata:
    if x:
        new_data.append(x)
rawdata = new_data

# Split into training and testing data, 50:50 ratio
datasize = len(rawdata)
trainarray = random.sample(range(0, datasize), int(datasize/2))
for i in trainarray:
    traindata.append(rawdata[i][0:3])
    trainvalues.append(rawdata[i][3])

for i in range(0, datasize):
    if i not in trainarray:
        testdata.append(rawdata[i][0:3])
        testvalues.append(rawdata[i][3])





print("TRAIN DATA:\n")
print(traindata)
print(trainvalues)
print("TEST DATA:\n")
print(testdata)
print(testvalues)

# train and test svm


clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='poly',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

clf.fit(traindata, trainvalues)

result = clf.predict(testdata)

# Print results

print("PREDICTION:\n")
print(len(result))
print("\n")
print(result)
print("ACTUAL:\n")
print(testvalues)
print("\n")
print(len(testvalues))
print("\n")


correct = 0
for a in range(0, len(testvalues)):
    if testvalues[a] == result[a]:
        correct += 1


print("PERCENT:\n")
print(correct / len(testvalues))