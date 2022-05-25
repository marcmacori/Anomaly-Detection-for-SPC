# Synthetic Data Creator in Python
import numpy as np

# Basic pattern definitions:
def normal(mu, sigma, numobs):
    return(mu + sigma * np.random.normal(size = (numobs)))

# Equation: y(t)=mu+r(t)*sigma+a*sin(2*pi*t/T)
# r(t): inevitable accidental fluctuation with Gaussian distribution
# a: amplitude of cyclic pattern
# T: period of cyclic pattern
def cyclic(mu, sigma, numobs):
    return(mu + sigma * np.random.normal(size = (numobs))\
        + np.random.uniform(1.5 * sigma, 2.5 * sigma, size\
        = (numobs)) * np.sin( 2 * np.arange(1, numobs + 1)\
         * np.pi / np.random.uniform(8, 32)))

# Equation: y(t)=mu+r(t)*sigma+d*(-1)^t
# d: degree of state departure
def systematic(mu, sigma, numobs):
    return(mu + sigma * np.random.normal(size = (numobs))\
        + np.random.uniform(1 * sigma, 3 * sigma, size\
        = (numobs)) * (-1) ** np.arange(1, numobs + 1))

# Equation: y(t)=mu+r(t)*sigma*[0.2;0.4]
def stratified(mu, sigma, numobs):
    return(mu + sigma * np.random.normal(size = (numobs))\
        * np.random.uniform(0.2 * sigma, 0.4 * sigma, size = (numobs)))

# Equation: y(t)=mu+r(t)*sigma+t*g
# g: gradient of upward trend
def ut(mu, sigma, numobs):
    return(mu + sigma * np.random.normal(size = (numobs))\
        + np.arange(1, numobs + 1)\
        * np.random.uniform(0.025 * sigma, 0.15 * sigma, size = (1)))

# Equation: y(t)=mu+r(t)*sigma-t*g
# g: gradient of downward trend
def dt(mu, sigma, numobs):
    return( mu + sigma * np.random.normal(size = (numobs))\
        - np.arange(1, numobs + 1)\
        * np.random.uniform(0.025 * sigma, 0.15 * sigma, size = (1)))

# Equation: y(t)=mu+r(t)*sigma+k*s
# k=1 if t>=P, else=0
# P: moment of shift
# s: amplitude of shift
def us(mu, sigma, numobs, start):
    a = start <= np.arange(1, numobs + 1)
    b = np.random.uniform(1 * sigma, 3 * sigma, size = (1, 1))
    return(mu + sigma * np.random.normal(size = (numobs))\
        + a * b)

# Equation: y(t)=mu+r(t)*sigma-k*s
# k=1 if t>=P, else=0
# P: moment of shift
# s: amplitude of shift
def ds(mu, sigma, numobs, start):
    a = start <= np.arange(1, numobs + 1)
    b = np.random.uniform(1 * sigma, 3 * sigma, size = (1, 1))
    return(mu + sigma * np.random.normal(size = (numobs))\
        - a * b)

# Joining anomly pattern with normal pattern in specific zones
def TSwithAnomaly(mu, sigma, numobs, anstart, anend, type):
    if (type == "cyclic"):
        a = np.concatenate((normal(mu, sigma, anstart)\
            , cyclic(mu, sigma, anend - anstart), normal(mu, sigma, numobs - anend)))
    elif(type == "systematic"):
        a = np.concatenate((normal(mu, sigma, anstart)\
            , systematic(mu, sigma, anend - anstart), normal(mu, sigma, numobs - anend)))
    elif(type == "stratified"):
        a = np.concatenate((normal(mu, sigma, anstart)\
            , stratified(mu, sigma, anend - anstart), normal(mu, sigma, numobs - anend)))
    elif(type == "ut"):
        a = np.concatenate((normal(mu, sigma, anstart)\
            , ut(mu, sigma, anend - anstart), normal(mu, sigma, numobs - anend)))
    elif(type == "dt"):
        a = np.concatenate((normal(mu, sigma, anstart)\
            , dt(mu, sigma, anend - anstart), normal(mu, sigma, numobs - anend)))
    return(a)