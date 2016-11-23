import random

import numpy as np
import sys

from numpy.linalg import norm
from numpy.ma import multiply, transpose, exp, divide, log, add, subtract


class LogisticRegression:

    def __init__( self, N):
        self.f = self.createF()
        self.w = np.array([0, 0, 0])
        self.N = N
        self.trainingData = [self.createDataPoint() for _ in range(N)]
        self.X = np.asmatrix(map(lambda data: data[0], self.trainingData))
        self.y = np.asarray(map(lambda data: data[1], self.trainingData))
        self.learningRate = 0.01
        self.epoch = 0
        self.runLogisticRegression()

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

    def findEout(self, numSamples = 1000):
        e_out = 0
        dataSamples = [self.createDataPoint() for _ in range(numSamples)]
        for x,y in dataSamples:
            e_out += log(1 + exp(-1 * multiply(y, np.dot(transpose(self.w), x))))
        e_out /= float(numSamples)
        return e_out

    def getEpoch(self):
        return self.epoch

    def runLogisticRegression(self):
        diff = 1
        while diff > 0.01:
            permutation = np.random.permutation(self.N)
            newWeights = self.w.copy()
            for i in permutation:
                x,y = self.trainingData[i]
                gradient = divide(multiply(-1.0, multiply(x, y)), (1.0 + exp(multiply(y,np.dot(transpose(self.w), x)))))
                newWeights = subtract(newWeights, multiply(self.learningRate, gradient))
            self.epoch += 1
            diff = norm(self.w - newWeights)
            self.w = newWeights

if __name__ == "__main__":
    numIterations = 100
    averageEin = 0
    averageEout = 0
    averageEpochs = 0
    for i in range(numIterations):
        logisticRegression = LogisticRegression(int(sys.argv[1]))
        averageEout += logisticRegression.findEout() / float(numIterations)
        averageEpochs += logisticRegression.getEpoch() / float(numIterations)
    print "Average Ein: " + str(averageEin)
    print "Average Eout: " + str(averageEout)
    print "Average Epochs: " + str(averageEpochs)
