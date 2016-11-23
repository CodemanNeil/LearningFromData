from numpy.random.mtrand import RandomState
from sklearn import svm, cross_validation

import numpy as np
from numpy import dot
from numpy.linalg import inv


class RegularizedLinearRegression:

    def __init__( self, regParam, positiveDigit, negativeDigits):
        self.positiveDigit = positiveDigit
        self.negativeDigits = negativeDigits
        train = self.transformData(np.loadtxt("features.train"))
        test = self.transformData(np.loadtxt("features.test"))
        self.Z_train = train[:, 0:3]
        self.y_train = train[:,3]
        self.Z_test = test[:, 0:3]
        self.y_test = test[:,3]
        self.w = dot(
                    dot(
                        inv(dot(self.Z_train.T, self.Z_train) + regParam * np.identity(self.Z_train.shape[1])),
                        self.Z_train.T),
                    self.y_train)


    def transformData(self, data):

        # First delete any rows that don't contain digit values either matching the positive digit
        # or in the list of negative digits.
        validDigits = self.negativeDigits[:]
        validDigits.append(self.positiveDigit)
        rowsToDelete = []
        for index, row in enumerate(data):
            if row[0] not in validDigits:
                rowsToDelete.append(index)

        validData = np.delete(data, rowsToDelete, 0)

        # Now transform the valid data into intensity, symmetry, +/- 1 (X, y)
        transform = lambda row: (1.0, row[1], row[2], 1.0 if row[0] == self.positiveDigit else -1.0)
        return np.apply_along_axis(transform, 1, validData)

    def findEin(self):
        numWrong = 0
        for index, x in enumerate(self.Z_train):
            y = np.sign(dot(self.w.T, x))
            if self.y_train[index] != y:
                numWrong += 1
        return float(numWrong) / float(len(self.y_train))

    def findEout(self):
        numWrong = 0
        for index, x in enumerate(self.Z_test):
            y = np.sign(dot(self.w.T, x))
            if self.y_test[index] != y:
                numWrong += 1
        return float(numWrong) / float(len(self.y_test))

class RegularizedNonLinearRegression:

    def __init__( self, regParam, positiveDigit, negativeDigits):
        self.positiveDigit = positiveDigit
        self.negativeDigits = negativeDigits
        train = self.transformData(np.loadtxt("features.train"))
        test = self.transformData(np.loadtxt("features.test"))
        self.Z_train = train[:, 0:6]
        self.y_train = train[:,6]
        self.Z_test = test[:, 0:6]
        self.y_test = test[:,6]
        self.w = dot(
                    dot(
                        inv(dot(self.Z_train.T, self.Z_train) + regParam * np.identity(self.Z_train.shape[1])),
                        self.Z_train.T),
                    self.y_train)


    def transformData(self, data):

        # First delete any rows that don't contain digit values either matching the positive digit
        # or in the list of negative digits.
        validDigits = self.negativeDigits[:]
        validDigits.append(self.positiveDigit)
        rowsToDelete = []
        for index, row in enumerate(data):
            if row[0] not in validDigits:
                rowsToDelete.append(index)

        validData = np.delete(data, rowsToDelete, 0)

        # Now transform the valid data into (X,y)
        return np.apply_along_axis(self.transform, 1, validData)

    def findEin(self):
        numWrong = 0
        for index, x in enumerate(self.Z_train):
            y = np.sign(dot(self.w.T, x))
            if self.y_train[index] != y:
                numWrong += 1
        return float(numWrong) / float(len(self.y_train))

    def findEout(self):
        numWrong = 0
        for index, x in enumerate(self.Z_test):
            y = np.sign(dot(self.w.T, x))
            if self.y_test[index] != y:
                numWrong += 1
        return float(numWrong) / float(len(self.y_test))

    def transform(self, dataRow):
        x1 = dataRow[1]
        x2 = dataRow[2]
        y = 1.0 if dataRow[0] == self.positiveDigit else -1.0
        return 1.0, x1, x2, x1*x2, x1**2, x2**2, y

if __name__ == "__main__":
    # numbers = range(10)
    # for positiveDigit in numbers:
    #     negativeDigits = list(numbers)
    #     negativeDigits.remove(positiveDigit)
    #     regLR = RegularizedLinearRegression(1, positiveDigit, negativeDigits)
    #     print "Reg. Linear Regression for ", positiveDigit, " vs All"
    #     print "Ein: ", regLR.findEin()
    #     print "Eout: ", regLR.findEout()
    #     print "\n"
    #     regNLR = RegularizedNonLinearRegression(1, positiveDigit, negativeDigits)
    #     print "Reg. Non Linear Regression for ", positiveDigit, " vs All"
    #     print "Ein: ", regNLR.findEin()
    #     print "Eout: ", regNLR.findEout()
    #     print "\n"
    #     print "\n"

    regParams = (0.01, 1)
    for regParam in regParams:
        regNLR = RegularizedNonLinearRegression(regParam, 1, [5])
        print "Reg. Non Linear Regression for 1 vs 5 and Regularization Param: ", regParam
        print "Ein: ", regNLR.findEin()
        print "Eout: ", regNLR.findEout()
        print "\n"
        print "\n"
