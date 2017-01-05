from decisiontree import extractUniqueLabels
from randomforest import trainRandomForest
from random import seed
from random import randrange
from random import shuffle
from csv import reader
from math import sqrt

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
	numTrees = 50
	trainingSet, testSet = partition(dataset, 0.8)
	labels = extractUniqueLabels(dataset)
	trainRandomForest(numTrees, dataset, trainingSet, testSet, labels)


	# numTrees = 50
	# Trees = []
	# accuracyAverage = 0;
	# for k in range(0, numTrees):
    #
    #
    #
	#
    #
	# 	tree = Dtree()
    #
	# 	labels = extractUniqueLabels(dataset)
	# 	#print("labels",labels)
    #
	# 	features = [x for x in range(len(dataset[0])-1)]
	# 	#print("features",features)
    #
	# 	tree.train(trainingSet,features)
    #
	# 	correct = 0
	# 	for testSample in testSet:
	# 		result = tree.classify(testSample)
	# 		if result == testSample[-1]:
	# 			correct += 1
    #
	# 	print(k)
	# 	accuracyAverage = accuracyAverage + (correct/len(testSet)*100);
	# 	Trees.append(tree)
    #
	# print("Accuracy Average", accuracyAverage/50,"%")

if __name__ == '__main__':
	main()