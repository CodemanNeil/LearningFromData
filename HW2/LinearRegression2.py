import random

import math
import numpy as np
import sys


class LinearRegression:

    def __init__( self, N):
        self.f = self.createF()
        self.w = np.array([0, 0, 0, 0, 0, 0])
        self.N = N
        self.trainingData = [self.createDataPoint() for _ in range(N)]
        self.X = np.asmatrix(map(lambda data: data[0], self.trainingData))
        self.y = np.asarray(map(lambda data: data[1], self.trainingData))
        self.w = runLinearRegression(self.X, self.y)

    def createF(self):
        return lambda x: np.sign(x[1] ** 2 + x[2] ** 2 - 0.6)

    def createRandomPoint(self):
        return random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)

    def createDataPoint(self):
        randomPoint = self.createRandomPoint()
        x = np.array([1, randomPoint[0], randomPoint[1], randomPoint[0] * randomPoint[1], randomPoint[0]**2, randomPoint[1]**2])
        y = self.f(x)
        if random.random() < 0.1:
            y *= -1
        return x, y

    def findEin(self):
        numWrong = 0
        for d in self.trainingData:
            x = d[0]
            y = np.sign(np.dot(self.w, x))
            if d[1] != y:
                numWrong += 1
        return float(numWrong) / float(self.N)

    def findEout(self, numSamples = 1000):
        numWrong = 0
        dataSamples = [self.createDataPoint() for _ in range(numSamples)]
        for d in dataSamples:
            x = d[0]
            y = np.sign(np.dot(self.w, x))
            if d[1] != y:
                numWrong += 1
        return float(numWrong) / float(numSamples)

    def getW(self):
        return np.squeeze(np.asarray(self.w))

def runLinearRegression(X, y):
    Xpinv = np.linalg.pinv(X)
    return np.dot(Xpinv, y)

if __name__ == "__main__":
    numIterations = 1000
    averageEin = 0
    averageEout = 0
    averageW = np.array([0.0] * 6)
    for i in range(numIterations):
        linearRegression = LinearRegression(int(sys.argv[1]))
        averageEin += linearRegression.findEin() / float(numIterations)
        averageEout += linearRegression.findEout() / float(numIterations)
        averageW += linearRegression.getW() / float(numIterations)
    print "Average Ein: " + str(averageEin)
    print "Average Eout: " + str(averageEout)
    print "Weight vector: "
    print averageW
