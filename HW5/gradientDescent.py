from math import exp


def gradientDescent():
    print "Gradient Descent"
    iter = 0
    error = 0
    stop = False
    u,v = (1.0,1.0)
    learningRate = .1
    while not stop:
        iter += 1
        d_du = 2.0 * (exp(v) + 2.0 * v * exp(-u)) * (u * exp(v) - 2.0 * v * exp(-u))
        d_dv = 2.0 * u**2.0 * exp(2.0 * v) - 4.0 * u * exp(v-u) - 4.0 * u * v * exp(v-u) + 8.0 * exp(-2.0 * u) * v
        u += learningRate * (-1.0 * d_du)
        v += learningRate * (-1.0 * d_dv)
        error = (u * exp(v) - 2.0 * v * exp(-u))**2.0
        if (error < 10**(-14)):
            stop = True
    print "Iterations: " + str(iter)
    print "(u,v): (" + str(u) + "," + str(v) + ")"
    print "Error: " + str(error)
    print "\n"

def coordinateDescent():
    print "Coordinate Descent"
    error = 0
    u,v = (1.0,1.0)
    learningRate = .1
    for i in range(15):
        d_du = 2.0 * (exp(v) + 2.0 * v * exp(-u)) * (u * exp(v) - 2.0 * v * exp(-u))
        u += learningRate * (-1.0 * d_du)

        d_dv = 2.0 * u**2.0 * exp(2.0 * v) - 4.0 * u * exp(v-u) - 4.0 * u * v * exp(v-u) + 8.0 * exp(-2.0 * u) * v
        v += learningRate * (-1.0 * d_dv)

        error = (u * exp(v) - 2.0 * v * exp(-u))**2.0

    print "(u,v): (" + str(u) + "," + str(v) + ")"
    print "Error: " + str(error)
    print "\n"

gradientDescent()
coordinateDescent()
