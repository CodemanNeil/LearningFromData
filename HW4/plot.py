import matplotlib.pyplot as plt
from numpy import linspace
from numpy.ma import divide, power, multiply, log, sqrt

n = linspace(0,10000,10000)
def vcBound(n):
    return sqrt(multiply(divide(8.0,n),log(multiply(4,divide(power(multiply(2,n),50), 0.05)))))
def rpBound(n):
    return sqrt(divide(multiply(2,log(multiply(multiply(2,n),power(n,50)))),n)) + sqrt(multiply(divide(2,n),log(divide(1,0.05)))) + divide(1,n)

def pvbBound(n):
    a = multiply(2,n)
    b = power(a,50)
    c = multiply(6,b)
    d = divide(c,0.05)
    e = log(d)
    f = divide(1.0,n)
    return divide(1.0, n) + sqrt(divide(1.0,power(n,2)) + multiply(f,e))

def devroyeBound(n):
    # Ran into overflow error performing naive calculation.  Had to decompose natural log components.
    return divide(1,(n - 2.0)) + sqrt(divide(1,power(n-2.0,2)) + multiply(divide(1,multiply(2,(n - 2.0))), log(4) + multiply(100, log(n)) - log(0.05)))

plt.plot(n,vcBound(n),'r')
plt.plot(n,rpBound(n),'b')
plt.plot(n,pvbBound(n),'g')
plt.plot(n,devroyeBound(n),'y')
plt.axis([0, 10000, 0, 1])
plt.show()