import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import re
import math

class Classifier:
    def __init__(self, numObjects, numFeatures, filePath):
        self.numObjects = numObjects
        self.numFeatures = numFeatures
        self.filePath = filePath
        self.data = self.readFile(self.filePath)
        self.setParams(self.data)
        self.train()
        #self.testData = self.readFile("yeast_test.txt")
        self.test(self.readFile("yeast_test.txt"))

    def readFile(self, filePath):
        self.file = open(filePath, encoding = 'utf-8')
        return self.parse()

    def parse(self):
        self.inputMatrix = [[str(digit) for digit in line.strip().split(' ')] for line in self.file]
        data = [[i for i in vector if len(i)>0] for vector in self.inputMatrix]
        data = np.asarray(data).astype(float)
        return data

    def setParams(self, data):
        self.minClass = int(np.amin(data[:,self.numFeatures - 1]))
        self.maxClass = int(np.amax(data[:,self.numFeatures - 1]))
        self.numClasses = self.maxClass - self.minClass        

    def train(self):
        #dimension 1: classes
        #dimension 2: attribute/feature
        #dimension 3: avg/stdev (col 1 = avg, col 2 = stdev)
        self.featureData = np.zeros((self.numClasses, self.numFeatures - 1, 2))
        for i in range(self.minClass, self.maxClass):
            featureSum = 0
            thisFeature = np.where(self.data[:,self.numFeatures - 1] == i)
            featureCount = len(self.data[thisFeature])
            for k in range(self.numFeatures - 1):
                featureSum = 0
                featureSum = self.data[thisFeature,k].sum()
                featureAvg = featureSum / featureCount
                featureStDev = np.std(self.data[thisFeature,k])
                if(featureStDev < 0.01):
                    featureStDev = 0.01

                self.featureData[i - self.minClass, k, 0] = featureAvg
                self.featureData[i - self.minClass, k, 1] = featureStDev
        self.printTrainingData()


    def printTrainingData(self):
        for i in range(self.numClasses - self.minClass):
            for j in range(self.numFeatures - 1):
                print(f"CLASS {i+1} and ATTRIBUTE {j+1}: average = ", round(self.featureData[i][j][0], 2), "; stdev = ", round(self.featureData[i][j][1], 2))

                      
    def gaussian(self, val, mean, stdev):
        exp = math.exp(-.5 * pow(((val - mean)/stdev),2))
        base = 1/(math.sqrt(2*math.pi) * stdev)
        ret = base * exp
        return ret


    def test(self, data):
        predictions = np.zeros((len(data), self.numClasses)) #number of objects, number of classes [10]
        totalVals = len(data)
        correctVals = 0
        accuracy = 0
        for i in range(len(data)):
            accuracy = 0
            for j in range(self.numClasses - self.minClass):
                predictions[i,j] = 1
                for k in range(self.numFeatures - 1):
                    predictions[i,j] *= self.gaussian(data[i][k],self.featureData[j][k][0], self.featureData[j][k][1])
            maxPred = np.argmax(predictions[i]) + 1
            realVal = data[i][self.numFeatures - 1]
            if(maxPred == realVal):
                accuracy = 1
                correctVals = correctVals + accuracy
            print(f"ID={i + 1}, predicted={maxPred}, true={int(realVal)}, accuracy=={accuracy}")
        print("Total percent accuracy: ", float(correctVals)/totalVals)


    


    
        

Classifier(484, 9, "yeast_training.txt")
