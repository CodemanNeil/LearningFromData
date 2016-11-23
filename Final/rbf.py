from random import uniform

from numpy.ma import dot, subtract
from sklearn import svm

import numpy as np
from numpy.core.umath import sign, sin, pi, log, exp, multiply
from numpy.linalg import norm, inv
from sklearn.cluster import KMeans
from sklearn.svm import SVC


class LearningAlgorithm:

    def __init__( self, gamma):
        self.gamma = gamma
        self.w = np.array([0, 0, 0])
        self.trainingData = [self.createDataPoint() for _ in range(100)]
        self.X = np.asmatrix(map(lambda data: data[0], self.trainingData))
        self.y = np.asarray(map(lambda data: data[1], self.trainingData))


    def createRandomPoint(self):
        return uniform(-1.0, 1.0), uniform(-1.0, 1.0)

    def createDataPoint(self):
        randomPoint = self.createRandomPoint()
        x = np.array([1, randomPoint[0], randomPoint[1]])
        return x, f(x[1], x[2])


class RBF(LearningAlgorithm):
    def __init__(self, gamma, k):
        LearningAlgorithm.__init__(self, gamma)
        self.kMeans = KMeans(n_clusters=k, init='random').fit(self.X).cluster_centers_
        self.Z = np.apply_along_axis(self.transformX, 1, self.X)
        self.w = dot(
                    dot(inv(dot(self.Z.T, self.Z)), self.Z.T),
                    self.y)

    def transformX(self, x):
        z = [1]
        for kMean in self.kMeans:
            z.append(exp((-self.gamma) * (norm(subtract(x,kMean)))**2))
        return z

    def findEin(self):
        numberWrong = 0
        for index, z in enumerate(self.Z):
            y_expected = self.y[index]
            y = sign(dot(z, self.w.T))
            if y != y_expected:
                numberWrong += 1
        return numberWrong / 100.0

    def findEout(self, numSamples = 1000):
        e_out = 0
        dataSamples = [self.createDataPoint() for _ in range(numSamples)]
        for x,y_expected in dataSamples:
            z = np.asarray(self.transformX(x))
            y = sign(dot(self.w.T,z))
            if y != y_expected:
                e_out += 1
        e_out /= float(numSamples)
        return e_out

class RbfSvm(LearningAlgorithm):
    def __init__(self, gamma):
        LearningAlgorithm.__init__(self, gamma)
        self.svm = SVC(C=float('inf'))
        self.svm.fit(self.X, self.y)

    def findEin(self):
        numberWrong = 0
        for index, x in enumerate(self.X):
            y_expected = self.y[index]
            y = self.svm.predict(x)
            if y != y_expected:
                numberWrong += 1
        return numberWrong / 100.0

    def findEout(self, numSamples = 1000):
        e_out = 0
        dataSamples = [self.createDataPoint() for _ in range(numSamples)]
        for x,y_expected in dataSamples:
            x = x.reshape(1,-1)
            y = self.svm.predict(x)
            if y != y_expected:
                e_out += 1
        e_out /= float(numSamples)
        return e_out


def f(x1, x2):
    return sign(x2 - x1 + (0.25 * sin(pi * x1)))

if __name__ == "__main__":
    K = [9,12]
    for k in K:
        numSvmNotSeperable = 0
        numRbfNotSeperable = 0
        numSvmBetterPerformance = 0
        for i in range(100):
            rbfSvm = RbfSvm(1.5)
            # if rbfSvm.findEin() > 0:
            #     numSvmNotSeperable += 1

            rbf = RBF(1.5, k)
            if rbf.findEin() > 0:
                numRbfNotSeperable +=1

            # if rbfSvm.findEout() < rbf.findEout():
            #     numSvmBetterPerformance += 1
        print "K: ", k
        print "% SVM not seperable: ", numSvmNotSeperable / 100.0 * 100
        print "% RBF not seperable: ", numRbfNotSeperable / 100.0 * 100
    #     print "% SVM better performance: ", numSvmBetterPerformance / 100.0 * 100

    # rbf12_ein_better = 0
    # rbf12_eout_better = 0
    # for i in range(100):
    #     rbf9 = RBF(1.5, 9)
    #     rbf12 = RBF(1.5, 12)
    #     if rbf12.findEin() < rbf9.findEin():
    #         rbf12_ein_better += 1
    #     if rbf12.findEout() < rbf9.findEout():
    #         rbf12_eout_better += 1
    # print "% RBF12 Ein Better: ", rbf12_ein_better / 100.0 * 100
    # print "% RBF12 Eout Better: ", rbf12_eout_better / 100.0 * 100

    # rbf2_ein_better = 0
    # rbf2_eout_better = 0
    # for i in range(1000):
    #     rbf1_5 = RBF(1.5, 9)
    #     rbf2 = RBF(2, 9)
    #     if rbf2.findEin() < rbf1_5.findEin():
    #         rbf2_ein_better += 1
    #     if rbf2.findEout() < rbf1_5.findEout():
    #         rbf2_eout_better += 1
    # print "% RBF12 Ein Better: ", rbf2_ein_better / 1000.0 * 100
    # print "% RBF12 Eout Better: ", rbf2_eout_better / 1000.0 * 100