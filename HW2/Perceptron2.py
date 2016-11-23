from random import uniform

import numpy as np
import sys

from LinearRegression import runLinearRegression


class Perceptron2:

    def __init__( self, N):
        self.f = self.createF()
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
        return uniform(-1.0, 1.0), uniform(-1.0, 1.0)

    def createDataPoint(self):
        randomPoint = self.createRandomPoint()
        x = np.array([1, randomPoint[0], randomPoint[1]])
        return x, self.f(x)

    def runPla(self):
        converged = False
        numIterations = 0
        while not converged:
            updateOccured = False
            for d in self.trainingData:
                x = d[0]
                y = np.sign(np.dot(self.w, x))
                if d[1] != y:
                    updateOccured = True
                    numIterations += 1
                    self.w = np.add(self.w, np.multiply((d[1] - y), x))
            if not updateOccured:
                converged = True
        return numIterations

    def checkAccuracy(self, numDataPoints = 10000):
        numWrong = 0
        data = [self.createDataPoint() for _ in range(numDataPoints)]
        for d in data:
            x = d[0]
            y = np.sign(np.dot(self.w, x))
            if d[1] != y:
                numWrong += 1
        return float(numWrong) / float(numDataPoints)


if __name__ == "__main__":
    averageIterations = 0
    averagePercentWrong = 0
    for i in range(1000):
        perceptron = Perceptron2(int(sys.argv[1]))
        numIterations = perceptron.runPla()
        averageIterations += numIterations
        percentWrong = perceptron.checkAccuracy()
        averagePercentWrong += percentWrong
    averageIterations = averageIterations / 1000
    averagePercentWrong = averagePercentWrong / 1000
    print "Percent Wrong: " + str(averagePercentWrong)
    print "Iterations: " + str(averageIterations)
