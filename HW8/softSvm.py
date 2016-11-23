from numpy.random.mtrand import RandomState
from sklearn import svm, cross_validation

import numpy as np


class SoftSvm:

    def __init__( self, C, Q, positiveDigit, negativeDigits):
        self.positiveDigit = positiveDigit
        self.negativeDigits = negativeDigits
        train = self.transformData(np.loadtxt("features.train"))
        test = self.transformData(np.loadtxt("features.test"))
        self.X_train = train[:,0:1]
        self.y_train = train[:,2]
        self.X_test = test[:,0:1]
        self.y_test = test[:,2]

        # self.svm = svm.SVC(C=C, degree=Q, kernel="poly", random_state=RandomState())
        self.svm = svm.SVC(C=C, degree=Q, kernel="rbf", random_state=RandomState())
        self.svm.fit(self.X_train, self.y_train)

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
        transform = lambda row: (row[1], row[2], 1.0 if row[0] == self.positiveDigit else -1.0)
        return np.apply_along_axis(transform, 1, validData)

    def findEin(self):
        numSamples = len(self.y_train)
        y_predicted = self.svm.predict(self.X_train)
        numWrong = numSamples - np.sum(y_predicted == self.y_train.T)
        return float(numWrong) / float(numSamples)

    def findEout(self):
        numSamples = len(self.y_test)
        y_predicted = self.svm.predict(self.X_test)
        numWrong = numSamples - np.sum(y_predicted == self.y_test.T)
        return float(numWrong) / float(numSamples)

    def getNumSupportVectors(self):
        return self.svm.n_support_


class SoftSvmWithCV:

    def __init__( self, C, Q, positiveDigit, negativeDigits):
        self.positiveDigit = positiveDigit
        self.negativeDigits = negativeDigits
        train = self.transformData(np.loadtxt("features.train"))
        test = self.transformData(np.loadtxt("features.test"))
        self.X_train = train[:,0:2]
        self.y_train = train[:,2]
        self.X_test = test[:,0:1]
        self.y_test = test[:,2]

        self.svm = svm.SVC(C=C, degree=Q, kernel="poly", random_state=RandomState())

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
        transform = lambda row: (row[1], row[2], 1.0 if row[0] == self.positiveDigit else -1.0)
        return np.apply_along_axis(transform, 1, validData)

    def getEcv(self):
        return 1.0 - cross_validation.cross_val_score(self.svm, self.X_train, self.y_train, cv=10, scoring='accuracy').mean()

if __name__ == "__main__":
    numbers = range(10)
    # for positiveDigit in numbers:
    #     negativeDigits = list(numbers)
    #     negativeDigits.remove(positiveDigit)
    #     softSvm = SoftSvm(0.01, 2, positiveDigit, negativeDigits)
    #     print "SVM for ", positiveDigit, " vs All"
    #     print "Num support vectors: ", np.sum(softSvm.getNumSupportVectors())
    #     print "Ein: ", softSvm.findEin()
    #     print "Eout: ", softSvm.findEout()
    #     print "\n"

    Q = [2,5]
    C = [0.0001, 0.001,0.01,0.1,1]

    # for c in C:
    #     for q in Q:
    #         softSvm = SoftSvm(c, q, 1, [5])
    #         print "SVM for C: ", c, ", Q: ", q
    #         print "Num support vectors: ", np.sum(softSvm.getNumSupportVectors())
    #         print "Ein: ", softSvm.findEin()
    #         print "Eout: ", softSvm.findEout()
    #         print "\n"

    for c in C:
        averageEcv = 0
        for i in range(100):
            crossValidationSvm = SoftSvmWithCV(c, 2, 1, [5])
            averageEcv += crossValidationSvm.getEcv() / 100.0
        print "SVM for C: ", c, ", Q: 2"
        print "Ecv: ", averageEcv
        print "\n"

    # C = [0.01, 1, 100, 10**4, 10**6]
    #
    # for c in C:
    #     softSvm = SoftSvm(c, 2, 1, [5])
    #     print "SVM for C: ", c
    #     print "Num support vectors: ", np.sum(softSvm.getNumSupportVectors())
    #     print "Ein: ", softSvm.findEin()
    #     print "Eout: ", softSvm.findEout()
    #     print "\n"
