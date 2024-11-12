# Ivan Chu
#1001593303

import sys
import numpy as np
import math
import random


class Tree:
    def __init__(self, attr=-1, thresh=-1, left=None, right=None, data=-1, gain=0):
        self.attr = attr
        self.thresh = thresh
        self.left = left
        self.right = right
        self.data = data
        self.gain = gain


def DTL_TopLevel(examples, pruning_thr, attrs):
    attributes = attrs
    default = calcDistribution(examples)[1]
    return DTL(examples, attributes, default, pruning_thr)


def DTL(examples, attributes, default, pruning_thr):
    if len(examples) < pruning_thr:
        return Tree(data=default)
    distribution_array = calcDistribution(examples)
    if 1 in distribution_array:
        return Tree(data=distribution_array)
    else:
        best_attribute, best_threshold, gain = choose_attribute(
            examples, attributes, option)
        tree = Tree(best_attribute, best_threshold, gain=gain)
        examples_left = [
            x for x in examples if x[best_attribute] < best_threshold]
        examples_right = [
            x for x in examples if x[best_attribute] >= best_threshold]
        tree.left = DTL(examples_left, attributes,
                        distribution_array, pruning_thr)
        tree.right = DTL(examples_right, attributes,
                         distribution_array, pruning_thr)
        return tree


def choose_attribute(examples, attributes, choose_type):
    if choose_type == "optimized":
        max_gain = best_attribute = best_threshold = -1
        for A in attributes:
            attribute_values = [x[A] for x in examples]
            L = min(attribute_values)
            M = max(attribute_values)
            for K in range(1, 51):
                threshold = L + K*(M-L)/51
                gain = info_gain(examples, A, threshold)
                if gain > max_gain:
                    max_gain = gain
                    best_attribute = A
                    best_threshold = threshold
        return (best_attribute, best_threshold, max_gain)
    elif choose_type == "randomized":
        max_gain = best_threshold = -1
        A = random.choice(attributes)
        attribute_values = [x[A] for x in examples]
        L = min(attribute_values)
        M = max(attribute_values)
        for K in range(1, 51):
            threshold = L + K*(M-L)/51
            gain = info_gain(examples, A, threshold)
            if gain > max_gain:
                max_gain = gain
                best_threshold = threshold
        return (A, best_threshold, max_gain)


def calcClasses(examples):
    classes = []
    for i in range(len(examples)):
        if examples[i][-1] not in classes:
            classes.append(examples[i][-1])
    temp = dict.fromkeys(classes, 0)
    counter = 0
    for i in sorted(temp.keys()):
        temp[i] = counter
        counter += 1
    return temp


def calcDistribution(examples):
    distribution_array = np.zeros(len(distribution))
    for i in range(len(examples)):
        distribution_array[distribution[examples[i][-1]]] += 1
    for i in range(len(distribution_array)):
        if len(examples) > 0:
            distribution_array[i] /= len(examples)
    return distribution_array


def info_gain(examples, attr, threshold):
    entropy = [0, 0, 0]
    examples_left = []
    examples_right = []
    numRows = len(examples)
    for i in examples:
        if i[attr] < threshold:
            examples_left.append(i)
        else:
            examples_right.append(i)
    dist = calcDistribution(examples)
    left = calcDistribution(examples_left)
    right = calcDistribution(examples_right)
    for num in dist:
        if num > 0:
            entropy[0] -= (num * np.log2(num))
    for num in left:
        if num > 0:
            entropy[1] -= (num * np.log2(num))
    for num in right:
        if num > 0:
            entropy[2] -= (num * np.log2(num))
    final_entropy = entropy[0] - ((len(examples_left) / numRows)
                                  * entropy[1]) - ((len(examples_right) / numRows) * entropy[2])
    return final_entropy


def predictVal(tree, input):
    if tree.left == None and tree.right == None:
        return tree.data
    if input[tree.attr] < tree.thresh:
        return predictVal(tree.left, input)
    else:
        return predictVal(tree.right, input)


def levelOrder(root):
    ret = []
    if not root:
        return ret
    q = [root]
    while q:
        ql = len(q)
        levelList = []
        while ql:
            ql -= 1
            node = q.pop(0)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
            levelList.append([node.attr, node.thresh, node.gain])
        ret.append(levelList)
    return ret


if len(sys.argv) < 5:
    print("Not enough arguments")
    exit(0)

option = sys.argv[3]
pruning_thr = int(sys.argv[4])


inFile = open(sys.argv[1], 'r')
lines = inFile.readlines()
trainData = []
for row in lines:
    temp = row.split()
    trainData.append(np.array(temp).astype(np.float64))

inFile = open(sys.argv[2], 'r')
lines = inFile.readlines()
testData = []
for row in lines:
    temp = row.split()
    testData.append(np.array(temp).astype(np.float64))

attributes = range(len(trainData[0][:-1]))


distribution = calcClasses(trainData)
trees = []

if option == "optimized":
    trees.append(DTL_TopLevel(trainData, pruning_thr, attributes))
elif option.isdigit():
    integer_greater_than_0 = int(option)
    if integer_greater_than_0 > 0:
        option = "randomized"
        for i in range(integer_greater_than_0):
            trees.append(DTL_TopLevel(trainData, pruning_thr, attributes))  
    else:
        print("Not a valid training option")
        exit(1)   
        
#############
# Above code is for user enter any int they want
#############
# Current commented ones are hard coded version
#elif option == "forest3":
#    option = "randomized"
#    for i in range(3):
#        trees.append(DTL_TopLevel(trainData, pruning_thr, attributes))
#elif option == "forest15":
#    option = "randomized"
#    for i in range(15):
#        trees.append(DTL_TopLevel(trainData, pruning_thr, attributes))
#############

else:
    print("Not a valid training option")
    exit(1)

for i in range(len(trees)):
    node = 1
    levels = levelOrder(trees[i])
    print(levels)
    for j in range(len(levels)):
        for k in range(len(levels[j])):
            print("tree={:2d}, node={:3d}, feature={:2d}, thr={:6.2f}, gain={:f}".format(i+1, node, levels[j][k][0], levels[j][k][1], levels[j][k][2]))
            node+=1


totalAcc = 0
for n in range(len(testData)):
    predictClass = []
    dist = []
    for i in range(len(trees)):
        dist.append(predictVal(trees[i], testData[n]))
    avgDist = np.zeros(len(dist[0]))
    for i in range(len(dist)):
        for j in range(len(dist[i])):
            avgDist[j] += dist[i][j]
    for i in range(len(avgDist)):
        avgDist[i] /= len(dist)

    max_index = []
    max_index.append(np.argmax(avgDist))

    for i in range(len(avgDist)):
        if avgDist[i] == avgDist[max_index[0]] and i not in max_index:
            max_index.append(i)
    for key, val in distribution.items():
        if val in max_index:
            predictClass.append(int(key))
    accuracy = 1 / \
        len(predictClass) if (int(testData[n][-1]) in predictClass) else 0
    totalAcc += accuracy
    print("ID={:5d}, predicted={:3d}, true={:3d}, accuracy={:4.2f}".format(
        n+1, int(predictClass[0]), int(testData[n][-1]), accuracy))
print("classification accuracy={:6.4f}".format(totalAcc/len(testData)))
