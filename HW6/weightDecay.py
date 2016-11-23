import numpy as np
import sys
from numpy import transpose as tp
from numpy.linalg import inv
from numpy import dot


class WeightDecay:

    def linearRegression(self, regParam):
        train = np.loadtxt("in.txt")
        test = np.loadtxt("out.txt")
        lin_train, self.y_train = np.array_split(train, 2, 1)
        lin_test, self.y_test = np.array_split(test, 2, 1)
        self.Z_train = np.apply_along_axis(self.nonLinearTransform, 1, lin_train)
        self.Z_test = np.apply_along_axis(self.nonLinearTransform, 1, lin_test)
        self.w = dot(
                    dot(
                        inv(dot(self.Z_train.T, self.Z_train) + regParam * np.identity(self.Z_train.shape[1])),
                        self.Z_train.T),
                    self.y_train)

    def nonLinearTransform(self, x):
        x1, x2 = x
        return 1.0, x1, x2, x1**2, x2**2, x1*x2, abs(x1 - x2), abs(x1 + x2)

    def findEin(self):
        numWrong = 0
        for index, x in enumerate(self.Z_train):
            y = np.sign(dot(tp(self.w), x))
            if self.y_train[index] != y:
                numWrong += 1
        return float(numWrong) / float(len(self.y_train))

    def findEout(self):
        numWrong = 0
        for index, x in enumerate(self.Z_test):
            y = np.sign(dot(tp(self.w), x))
            if self.y_test[index] != y:
                numWrong += 1
        return float(numWrong) / float(len(self.y_test))

if __name__ == "__main__":
    weightDecay = WeightDecay()
    weightDecay.linearRegression(10**(int(sys.argv[1])))
    print "Ein: ", weightDecay.findEin()
    print "Eout: ", weightDecay.findEout()