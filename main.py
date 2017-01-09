from randomforest import trainRandomForest
from random import seed
from random import randrange
from random import shuffle
from csv import reader
from math import sqrt
from decisiontree import Dtree
from math import log

# Load a CSV file
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
    numTrees = 100
    results = []
    numFeatures = 1
    numFeatures2 = round(log(totalNumFeatures+1, 2))

    for i in range(100):
        trainingSet, testSet = partition(dataset, 0.9)
        result1 = trainRandomForest(numTrees, dataset, trainingSet, testSet, 1)
        result2 = trainRandomForest(numTrees, dataset, trainingSet, testSet, numFeatures2)
        print(result1, result2)
        if result1 < result2:
            results.append(result2)
            print('run: ', i, 'accuracy: ', result2)
        else:
            results.append(result1)
            print('run: ', i, 'accuracy: ', result1)

#         results.append(result1)
#         print('run: ', i, 'accuracy: ', result1)

    accuracy = sum(results) / 100
    print("accuracy: ", accuracy, "%")

if __name__ == '__main__':
        main()
