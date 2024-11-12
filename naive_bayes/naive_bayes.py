#Ivan Chu
#1001593303

import sys
import csv
import numpy as np
import math

if len(sys.argv) < 3:
    print("Not enough arguments")
    exit(0)

readFile = open(sys.argv[1], 'r')
lines = readFile.readlines()
trainData = []
for row in lines:
    trainData.append(row.split())

readFile = open(sys.argv[2], 'r')
lines = readFile.readlines()
testData = []
for row in lines:
    testData.append(row.split())

# Format data into dictionaries and convert to floats
totalTrain = len(trainData)
totalTest = len(testData)

# seperate train based on class
trainClasses = dict()
for i in range(len(trainData)):
    vector = trainData[i]
    class_value = int(vector[-1])
    if (class_value not in trainClasses):
        trainClasses[class_value] = list()
    trainClasses[class_value].append(vector[:-1])

# convert train arrays to floats
for i in trainClasses:
    for j in range(len(trainClasses[i])):
        trainClasses[i][j] = np.array(trainClasses[i][j]).astype(np.float64)

# convert test arrays to floats
for i in range(len(testData)):
    testData[i] = np.array(testData[i]).astype(np.float64)


# prep mean and std matrices
numClass = len(trainClasses)
numAttr = len(next(iter(trainClasses.values()))[0])
trainMean = dict()
trainSTD = dict()

# calc mean and std for each class and attr
for i in trainClasses:
    tempMean = tempSTD = [0]*numAttr

    # calc mean
    for j in range(len(trainClasses[i])):
        tempMean += trainClasses[i][j]
    tempMean /= len(trainClasses[i])
    trainMean[i] = tempMean

    # calc std
    for j in range(len(trainClasses[i])):
        tempSTD += ((trainClasses[i][j] - trainMean[i])**2)
    tempSTD = (tempSTD/(len(trainClasses[i])-1))**(1/2)
    trainSTD[i] = tempSTD

# set to 0.01 if smaller
for i in trainSTD:
    for j in range(len(trainSTD[i])):
        if trainSTD[i][j] < 0.01:
            trainSTD[i][j] = 0.01

# calc class probabilities
classProb = dict()
for i in trainClasses:
    classProb[i] = len(trainClasses[i])/totalTrain

# print train stats
for i in sorted(trainMean.keys()):
    for j in range(len(trainMean[i])):
        print("Class {:d}, attribute {:d}, mean = {:0.2f}, std = {:0.2f}".format(
            i, j+1, trainMean[i][j], trainSTD[i][j]))

totalAcc = 0
for i in range(len(testData)):
    gaussian = dict()
    for j in trainMean:
        gaussian[j] = 1
    predict = (0, 0)
    accuracy = 1

    for j in trainMean:
        for k in range(len(testData[i])-1):
            # generate p(x|c) for each attribute
            gaussian[j] *= ((1/(trainSTD[j][k] * math.sqrt(2 * math.pi))) *
                            math.exp(-((testData[i][k] - trainMean[j][k])**2)/(2*trainSTD[j][k]**2)))
        gaussian[j] *= classProb[j]

    # find max prob using bayes rule
    for j in trainMean:
        if (gaussian[j]/sum(gaussian.values())) == predict[1]:
            accuracy += 1
        elif (gaussian[j]/sum(gaussian.values())) > predict[1]:
            predict = (j, gaussian[j]/sum(gaussian.values()))
            accuracy = 1

    # set accuracy of guesses
    accuracy = 1 if (int(predict[0]) == int(testData[i][-1])) else 0
    totalAcc += accuracy

    # print statistics
    print("ID={:5d}, predicted={:3d}, probability = {:0.4f}, true={:3d}, accuracy={:4.2f}".format(
        i+1, predict[0], predict[1], int(testData[i][-1]), accuracy))

print("classification accuracy={:6.4f}".format(totalAcc/len(testData)))
