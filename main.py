from decisiontree import extractUniqueLabels
from randomforest import trainRandomForest
from random import seed
from random import randrange
from random import shuffle
from csv import reader
from math import sqrt

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def partition(dataset, fraction):
    breakPoint = int(len(dataset) * fraction)
    shuffle(dataset)
    return dataset[:breakPoint], dataset[breakPoint:]


def main():
    dataset = load_csv("datasets/sonar.all-data.csv")
    labels = extractUniqueLabels(dataset)

    numTrees = 100
    trainingSet, testSet = partition(dataset, 0.8)

    results = []
    for i in range(100):
        print('run: ', i)
        results.append(trainRandomForest(numTrees, dataset, trainingSet, testSet, labels))

    accuracy = sum(results) / 100
    print("accuracy: ", accuracy, "%")


if __name__ == '__main__':
        main()
