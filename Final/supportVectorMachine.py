import numpy as np
from sklearn import svm


class SupportVectorMachine:

    def __init__( self):
        self.X, self.y = self.createData()

        self.svm = svm.SVC(C=10**15, kernel="poly", degree=2)
        self.svm.fit(self.X, self.y)

    def createData(self):
        X = np.asarray([[1.0, 1.0,0],[1.0, 0,1.0],[1.0, 0,-1.0],[1.0, -1.0,0],[1.0, 0,2.0],[1.0, 0,-2.0],[1.0, -2.0,0]])
        y = np.asarray([-1.0,-1.0,-1.0,1.0,1.0,1.0,1.0])
        return X,y

    def getWeights(self):
        return self.svm._get_coef()

    def getNumSupportVectors(self):
        return self.svm.n_support_

if __name__ == "__main__":
    supportVectorMachine = SupportVectorMachine()
    print "Weights: ", supportVectorMachine.getNumSupportVectors()
