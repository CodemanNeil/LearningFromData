import random

import numpy as np
import sys


class LinearRegression:

    def __init__( self, N):
        self.f = self.createF()
        self.w = np.array([0, 0, 0])
        self.N = N
        self.trainingData = [self.createDataPoint() for _ in range(N)]
        self.X = np.asmatrix(map(lambda data: data[0], self.trainingData))
        self.y = np.asarray(map(lambda data: data[1], self.trainingData))
        self.w = runLinearRegression(self.X, self.y)

    def createF(self):
        randomPoint1 = self.createRandomPoint()
        randomPoint2 = self.createRandomPoint()

        # We have two points of (x1, x2).
        # Replace y with x2 in y = mx+b => x2 = m(x1) + b => m(x1) - x2 + b = 0 => b + m(x1) + -1(x2) = 0
        # So the true weight vector for f where wTx = 0 will have the form [b, m, -1]
        m = (randomPoint2[1] - randomPoint1[1]) / (randomPoint2[0] - randomPoint1[0])
        b = randomPoint1[1] - m * randomPoint1[0]
        w = np.array([b, m, -1])
        return lambda x: np.sign(np.dot(x, w))

    def createRandomPoint(self):
        return random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)

    def createDataPoint(self):
        randomPoint = self.createRandomPoint()
        x = np.array([1, randomPoint[0], randomPoint[1]])
        return x, self.f(x)

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

def runLinearRegression(X, y):
    Xpinv = np.linalg.pinv(X)
    return np.dot(Xpinv, y)

if __name__ == "__main__":
    numIterations = 1000
    averageEin = 0
    averageEout = 0
    for i in range(numIterations):
        linearRegression = LinearRegression(int(sys.argv[1]))
        averageEin += linearRegression.findEin() / float(numIterations)
        averageEout += linearRegression.findEout() / float(numIterations)
    print "Average Ein: " + str(averageEin)
    print "Average Eout: " + str(averageEout)
