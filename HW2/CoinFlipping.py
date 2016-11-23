import random


class CoinFlipping:

    def __init__(self, numCoins = 1000, numFlips = 10):
        self.numCoins = numCoins
        self.numFlips = numFlips
        self.headFrequency = [0] * self.numCoins
        self.flipCoins()

    def flipCoins(self):
        for coinNumber in range(self.numCoins):
            for flipNumber in range(self.numFlips):
                # Heads
                if random.random() > 0.5:
                    self.headFrequency[coinNumber] += 1

    def getV1(self):
        return self.headFrequency[0] / float(self.numFlips)

    def getVRand(self):
        return random.choice(self.headFrequency) / float(self.numFlips)

    def getVMin(self):
        return min(self.headFrequency) / float(self.numFlips)

if __name__ == "__main__":
    numIterations = 100000
    averageV1 = 0
    averageVRand = 0
    averageVMin = 0

    for i in range(numIterations):
        coinFlipping = CoinFlipping()
        averageV1 += coinFlipping.getV1()
        averageVRand += coinFlipping.getVRand()
        averageVMin += coinFlipping.getVMin()

    averageV1 /= float(numIterations)
    averageVRand /= float(numIterations)
    averageVMin /= float(numIterations)

    print "Average V1: " + str(averageV1)

    print "Average Vmin: " + str(averageVMin)

    print "Average Vrand: " + str(averageVRand)