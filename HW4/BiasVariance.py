import random

from math import sin, pi


def createRandomPoint():
    x = random.uniform(-1.0, 1.0)
    return x, f(x)

def f(x):
    return sin(pi * x)

def getSlope():
    x1,y1 = createRandomPoint()
    x2,y2 = createRandomPoint()
    return (x1*y1 + x2*y2)/(x1**2 + x2**2)

def getB():
    x1,y1 = createRandomPoint()
    x2,y2 = createRandomPoint()
    return (y2 - y1)/2

#def getG(slope):
    #return lambda x: x * slope

def getG(b):
    return lambda x: b

def getGbar():
    averageB = 0
    numIterations = 1000000
    for _ in range(numIterations):
        averageB += getB() / float(numIterations)
    print "gBar: " + str(averageB)
    return getG(averageB)

def getBiasAndVariance(gBar):
    averageBias = 0
    averageVariance = 0
    numIterations = 100000
    for _ in range(numIterations):
        x1,y1 = createRandomPoint()
        averageBias += ((gBar(x1) - y1)**2)/float(numIterations)
        g = getG(getB())
        averageVariance += ((gBar(x1) - g(x1))**2)/float(numIterations)
    print "Bias: " + str(averageBias)
    print "Variance: " + str(averageVariance)



gBar = getGbar()
getBiasAndVariance(gBar)

