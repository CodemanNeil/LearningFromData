from random import uniform
from sklearn import svm

import numpy as np

class SupportVectorMachine:

    def __init__( self, N):
        self.f = self.createF()
        self.w = np.array([0, 0, 0])
        self.N = N
        dataIsValid = False
        while not dataIsValid:
            self.data = [self.createDataPoint() for _ in range(N)]
            self.X = np.asmatrix(map(lambda (x,y): x, self.data))
            self.y = np.asarray(map(lambda (x,y): y, self.data))
            if abs(sum(np.nditer(self.y))) < N:
                dataIsValid = True

        self.svm = svm.SVC(C=float("inf"), kernel="linear")
        self.svm.fit(self.X, self.y)

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

    def checkAccuracy(self, numDataPoints = 10000):
        data = [self.createDataPoint() for _ in range(numDataPoints)]
        X = np.asmatrix(map(lambda (x,y): x, data))
        y = np.asarray(map(lambda (x,y): y, data))
        y_test = self.svm.predict(X)
        numWrong = numDataPoints - np.sum(y == y_test)
        return float(numWrong) / float(numDataPoints)

    def getNumSupportVectors(self):
        return self.svm.n_support_


class Perceptron:

    def __init__( self, N):
        self.f = self.createF()
        self.w = np.array([0, 0, 0])
        self.N = N
        dataIsValid = False
        while not dataIsValid:
            self.data = [self.createDataPoint() for _ in range(N)]
            if abs(sum(map(lambda data: data[1], self.data))) < N:
                dataIsValid = True
        self.runPla()

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
            for d in self.data:
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
    n_values = 10,100
    for n in n_values:
        svmBetter = 0
        numSupportVectors = 0
        for i in range(1000):
            supportVectorMachine = SupportVectorMachine(n)
            perceptron = Perceptron(n)
            if supportVectorMachine.checkAccuracy() <= perceptron.checkAccuracy():
                svmBetter += 1
            numSupportVectors += supportVectorMachine.getNumSupportVectors()
            if i % 100 == 0:
                print "Iteration: ", i
        print "N: ", n, ", SVM better: ", svmBetter / 1000.0 * 100.0, "%."
        print "N: ", n, ", SVM support vectors: ", numSupportVectors / 1000.0
