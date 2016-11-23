import numpy as np
from numpy import transpose as tp
from numpy import dot


class Validation:

    def __init__(self, k, train_size):
        self.k = k
        self.train_size = train_size

    def linearRegression(self):
        input = np.loadtxt("in.txt")
        train = input[:self.train_size]
        validation = input[self.train_size:]
        test = np.loadtxt("out.txt")
        lin_train, self.y_train = np.array_split(train, 2, 1)
        lin_validation, self.y_validation = np.array_split(validation, 2, 1)
        lin_test, self.y_test = np.array_split(test, 2, 1)
        self.Z_train = np.apply_along_axis(self.nonLinearTransform, 1, lin_train)
        self.Z_validation = np.apply_along_axis(self.nonLinearTransform, 1, lin_validation)
        self.Z_test = np.apply_along_axis(self.nonLinearTransform, 1, lin_test)
        self.w = np.dot(np.linalg.pinv(self.Z_train), self.y_train)

    def nonLinearTransform(self, x):
        x1, x2 = x
        transforms = 1.0, x1, x2, x1**2, x2**2, x1*x2, abs(x1 - x2), abs(x1 + x2)
        return transforms[:self.k+1]

    def findEin(self):
        numWrong = 0
        for index, x in enumerate(self.Z_train):
            y = np.sign(dot(tp(self.w), x))
            if self.y_train[index] != y:
                numWrong += 1
        return float(numWrong) / float(len(self.y_train))

    def findEval(self):
        numWrong = 0
        for index, x in enumerate(self.Z_validation):
            y = np.sign(dot(tp(self.w), x))
            if self.y_validation[index] != y:
                numWrong += 1
        return float(numWrong) / float(len(self.y_validation))

    def findEout(self):
        numWrong = 0
        for index, x in enumerate(self.Z_test):
            y = np.sign(dot(tp(self.w), x))
            if self.y_test[index] != y:
                numWrong += 1
        return float(numWrong) / float(len(self.y_test))

if __name__ == "__main__":
    train_sizes = (10,25)
    k_values = (3,4,5,6,7)
    for train_size in train_sizes:
        for k in k_values:
            validation = Validation(k, train_size)
            validation.linearRegression()
            print "K: ", k, ", trainingSize: ", train_size, "Ein: ", validation.findEin()
            print "K: ", k, ", trainingSize: ", train_size, "Eval: ", validation.findEval()
            print "K: ", k, ", trainingSize: ", train_size, "Eout: ", validation.findEout()
            print "\n"