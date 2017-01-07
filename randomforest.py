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
def getSubSamples(data, ratio=1):
    sample = list()
    n_sample = round(len(data) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(data))
        sample.append(data[index])
    return sample

#Returns the majority vote for a specific sample
def getForestMajorityVote(forest, sample):
    results = dict()
    for tree in forest:
        result = tree.classify(sample)
        if result in results:
            results[result] += 1
        else:
            results[result] = 1
    return max(results, key=lambda x: results[x])

#Returns the accuracy of a single Decision Tree
def getForestAccuracy(forest, testSet):
    correct = 0
    for testSample in testSet:
        result = getForestMajorityVote(forest, testSample)
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

def trainRandomForest(numTrees, dataset, trainingSet, testSet, numFeatures):
    Trees = []
    subSets = []
    accuracyAverage = 0;

    #Grow all the trees in the forest
    for k in range(0, numTrees):
        tree = Dtree()
        features = [x for x in range(len(dataset[0]) - 1)]
        subSet = getSubSamples(trainingSet, 0.5)
#         subSets.append(subSet)
        tree.train(subSet, features, numFeatures)
        Trees.append(tree)

#     return outOfBagEstimate(dataset, subSets, Trees)
    return getForestAccuracy(Trees, testSet)

def outOfBagEstimate(dataset, subSets, trees):
    numSamples = len(dataset)
    numTrees = len(trees)
    correct = 0

    for sample in dataset:
        for i in range(numTrees):
            predictions = []
            if sample not in subSets[i]:
                predictions.append(trees[i].classify(sample))

        if len(predictions) == 0:
            numSamples -= 1
        else:
            result = max(set(predictions), key=predictions.count)
            if result == sample[-1]:
                correct += 1

    return (correct / numSamples * 100);

