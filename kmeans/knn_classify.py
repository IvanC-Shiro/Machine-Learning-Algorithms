# Ivan Chu
# 1001593303

import sys
import numpy as np
import math

def normalize():
    means = np.mean(trainData, axis=0)
    stds = np.std(trainData, axis=0, ddof=1)
    for i in range(len(stds)-1):
        if stds[i] == 0:
            stds[i] = 1
    for i in range(len(trainData)):
        for j in range(len(means)-1):
            trainData[i][j] = (trainData[i][j] - means[j])/stds[j]
    for i in range(len(testData)):
        for j in range(len(means)-1):
            testData[i][j] = (testData[i][j] - means[j])/stds[j]

def L_2(v1,v2):
    summation = 0
    for i in range(len(v1)-1):
        summation += (v1[i]- v2[i])**2
    return math.sqrt(summation)

def Sort(sub_li): 
    sub_li.sort(key = lambda x: x[0]) 
    return sub_li 

if len(sys.argv) < 4:
    print("Error, Not enough arguments")
    exit(0)

classes = []
k = int(sys.argv[3])

inFile = open(sys.argv[1], 'r')
lines = inFile.readlines()
trainData = []
for row in lines:
    temp = row.split()
    trainData.append(np.array(temp).astype(np.float64))
    if int(temp[-1]) not in classes:
        classes.append(int(temp[-1]))

inFile = open(sys.argv[2], 'r')
lines = inFile.readlines()
testData = []
for row in lines:
    temp = row.split()
    testData.append(np.array(temp).astype(np.float64))

normalize()

totalAcc = 0
for n in range(len(testData)):
    distances = []
    for i in range(len(trainData)):
        distances.append([L_2(trainData[i], testData[n]), trainData[i][-1]])
    Sort(distances)
    k_nearest_class = []
    for i in range(k):
        k_nearest_class.append(int(distances[i][1]))
    max_count = 0
    predictClass = [0]
    for i in classes:
        counts = k_nearest_class.count(i)
        if counts > max_count:
            max_count = counts
            predictClass.clear()
            predictClass.append(i)
        elif counts == max_count:
            predictClass.append(i)
    accuracy = 1/len(predictClass) if int(testData[n][-1] in predictClass) else 0
    totalAcc += accuracy
    print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}".format(n+1, int(predictClass[0]), int(testData[n][-1]), accuracy))
print("classification accuracy={:6.4f}".format(totalAcc/len(testData)))
