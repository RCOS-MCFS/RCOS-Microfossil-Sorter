# RCOS MCFS
# Testing use of scikit-learn's svm function

import numpy as np
#from sklearn import svm
from sklearn.svm import SVC
import csv
import sys
import random

# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])
# from sklearn.svm import SVC
# clf = SVC()
# clf.fit(X, y)
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)
# print(clf.predict([[-0.8, -1]]))

rawdata = [] # 1st bone data, 2nd rock data
traindata = []
trainvalues = []
testdata = []
testvalues = []
testresults = [] # Output


# Read in training data
def load_images(filename):
    print(filename + "\n")
    #print(os.path.join(path, filename) + "\n")
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            rawdata.append(row)



print(sys.argv[1] + "\n")
load_images(sys.argv[1])
print("Finished 1\n")


# Split into training and testing data, 20:80 ratio
datasize = len(rawdata);
trainarray = random.sample(range(0, datasize), datasize/5)
for i in trainarray:
    traindata.append(rawdata[i][0, 2])
    trainvalues.append(rawdata[i][3])

for i in range(0, datasize):
    if i not in trainarray:
        testdata.append(rawdata[i][0, 2])
        testvalues.append(rawdata[i][3])

print("Finished 2\n")

# train svm

clf = SVC()
clf.fit(traindata, trainvalues)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
     decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
     max_iter=-1, probability=False, random_state=None, shrinking=True,
     tol=0.001, verbose=False)

print(clf.predict(testdata))

