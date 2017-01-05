import math
import numpy
from decisiontree import Dtree
from random import seed
from random import randrange
from random import shuffle
from csv import reader
from math import sqrt
from numpy import random
from csv import reader


#Returns a subset (with replacement) of the data samples
def getSubSamples(data):
    # size = int(math.sqrt(len(data)))
    # dataindex = range(0, len(data)-1)
    # subindex = random.choice(dataindex, size, True)
    # sub = []
    # for index in subindex:
    #     sub.append(data[index])
    # return sub
    return data #above code should work, has been deactivated for testing


def getLabelIndex(labels, result):
    index = 0
    for label in labels:
            if(label == result):
                return index
            index +=1


#Returns the majority vote for a specific sample
def getForestMajorityVote(forest, labels, sample):
    labelcount = numpy.zeros(len(labels))

    for tree in forest:
        result = tree.classify(sample)
        index = getLabelIndex(labels, result)
        labelcount[index] += 1

    biggestLabelIndex = -99
    for index in range(0, len(labels)-1):
        if(biggestLabelIndex == -99):
            biggestLabelIndex = index
        if(labelcount[index] > labelcount[biggestLabelIndex]):
            biggestLabelIndex = index
    return labels[biggestLabelIndex]



#Returns the accuracy of a single Decision Tree
def getForestAccuracy(forest, testSet, labels):
    correct = 0
    for testSample in testSet:
        result = getForestMajorityVote(forest, labels, testSample)
        if result == testSample[-1]:
            correct += 1
    return (correct / len(testSet) * 100);



#Returns the accuracy of a single Decision Tree
def getTreeAccuracy(tree, testSet):
    correct = 0
    for testSample in testSet:
        result = tree.classify(testSample)
        if result == testSample[-1]:
            correct += 1
    return (correct / len(testSet) * 100);



def trainRandomForest(numTrees, dataset, trainingSet, testSet, labels):

    print("labels",labels)
    Trees = []
    accuracyAverage = 0;

    #Grow all the trees in the forest
    for k in range(0, numTrees):
        tree = Dtree()
        features = [x for x in range(len(dataset[0]) - 1)]

        tree.train(getSubSamples(trainingSet), features)
        #tree.train(trainingSet, features)
        Trees.append(tree)

    accuracy = getForestAccuracy(Trees, testSet, labels)
    print("Accuracy ", accuracy, "%")




