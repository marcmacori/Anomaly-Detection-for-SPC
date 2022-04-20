# Synthetic Data Creator in Python
import numpy as np
import pandas as pd 

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
         * np.pi / 16))

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
        * np.random.uniform(0.05 * sigma, 0.25 * sigma, size = (numobs)))

# Equation: y(t)=mu+r(t)*sigma-t*g
# g: gradient of downward trend
def dt(mu, sigma, numobs):
    return( mu + sigma * np.random.normal(size = (numobs))\
        - np.arange(1, numobs + 1)\
        * np.random.uniform(0.05 * sigma, 0.25 * sigma, size = (numobs)))

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

# creation of standard dataset: 280 time series with anomalies starting from point 20 and 
# starting with a 1 position delay for each new time series up until 60. 
# This is repeated for each type of anomaly. All points are labeled as in a corresponding matrix
# O corresponding to non-anomaly and 1 correspoding to anomaly
def DataSetCreation(mu, sigma, numobs):
    Xs, Ys = [], [],
    for i in range(40):
        X = TSwithAnomaly(mu, sigma, numobs, 20+i, numobs, "cyclic")
        Y = np.concatenate((np.zeros(20+i), np.ones(40-i)))
        Xs.append(X)
        Ys.append(Y)
    for i in range(40):
        X = TSwithAnomaly(mu, sigma, numobs, 20+i, numobs, "systematic")
        Y = np.concatenate((np.zeros(20+i), np.ones(40-i)))
        Xs.append(X)
        Ys.append(Y)
    for i in range(40):
        X = TSwithAnomaly(mu, sigma, numobs, 20+i, numobs, "stratified")
        Y = np.concatenate((np.zeros(20+i), np.ones(40-i)))
        Xs.append(X)
        Ys.append(Y)
    for i in range(40):
        X = TSwithAnomaly(mu, sigma, numobs, 20+i, numobs, "ut")
        Y = np.concatenate((np.zeros(20+i), np.ones(40-i)))
        Xs.append(X)
        Ys.append(Y)
    for i in range(40):
        X = TSwithAnomaly(mu, sigma, numobs, 20+i, numobs, "dt")
        Y = np.concatenate((np.zeros(20+i), np.ones(40-i)))
        Xs.append(X)
        Ys.append(Y)
    for i in range(40):
        X = np.transpose(us(mu,sigma,numobs,20+i))
        Y = np.concatenate((np.zeros(20+i), np.ones(40-i)))
        Xs.append(np.reshape(X, (60,)))
        Ys.append(Y)
    for i in range(40):
        X = np.transpose(ds(mu,sigma,numobs,20+i))
        Y = np.concatenate((np.zeros(20+i), np.ones(40-i)))
        Xs.append(np.reshape(X, (60,)))
        Ys.append(Y)
    return np.array(Xs), np.array(Ys) 

# Repeating the process to have 5 time series where the anomaly starts in the same position
TS1, TS_Classification1 = DataSetCreation(10, 1, 60)
TS2, TS_Classification2 = DataSetCreation(10, 1, 60)
TS3, TS_Classification3 = DataSetCreation(10, 1, 60)
TS4, TS_Classification4 = DataSetCreation(10, 1, 60)
TS5, TS_Classification5 = DataSetCreation(10, 1, 60)

# Final dataset
DataSet1 = np.concatenate((TS1, TS2, TS3, TS4, TS5))
DataSet1_Classification= np.concatenate((TS_Classification1, TS_Classification2,\
    TS_Classification3, TS_Classification4, TS_Classification5 ))

# Creating a dataframe and saving .csv
Df1= pd.DataFrame(DataSet1)
Df1_Classification=pd.DataFrame(DataSet1_Classification)
Df1.to_csv('TimeSeries1.csv')
Df1_Classification.to_csv('TimeSeries1_Classification.csv')

# Creating Dataset number 2. A multifeature dataset, with 2800 points and anomalies
# starting from point 1000. Each point is labeled as an anomaly or not in a 
# correspoinding vector with the same logic as above
def DataSet2Creation(len):
    
    mu = np.random.uniform(1, 50, size=1)
    sg = np.random.uniform(mu/20, mu/5, size=1)
    zmil = np.zeros(1000)

    a = [normal(mu, sg, 1000), zmil]

    Xs = a[0]
    Ys = a[1]

    while (Ys.shape[0] < len):
        
        lonn = np.random.randint(5, 200)
        lono = np.random.randint(5, 200)
        z = np.zeros(lonn)
        un = np.ones(lono)

        b = [normal(mu, sg, lonn), z]
        c = [cyclic(mu, sg, lono), un]
        d = [systematic(mu, sg, lono), un]
        e = [stratified(mu, sg, lono), un]
        f =[ut(mu, sg, lono), un]
        g = [dt(mu, sg, lono), un]
        h = [np.reshape(us(mu, sg, lono, 1), (lono,)), un]
        i = [np.reshape(ds(mu, sg, lono, 1), (lono,)), un]

        rand = np.random.permutation([b, b, b, b, b, c, d, e, f, g, h, i])
        Xs = np.concatenate((Xs, rand[0, 0]))
        Ys = np.concatenate((Ys, rand[0, 1]))

    if (Xs.shape[0] < len or Ys.shape[0] < len):
        Xs = np.concatenate((Xs, normal(mu, sg, len-Xs.shape[0])))
        Ys = np.concatenate((Ys, np.zeros(len-Xs.shape[0])))
    elif (Xs.shape[0] > len or Ys.shape[0] > len):
        Xs = Xs[0:len]
        Ys = Ys[0:len]

    return np.array(Xs), np.array(Ys)

a, b = DataSet2Creation(2800)
c, d = DataSet2Creation(2800)
e, f = DataSet2Creation(2800)
g, h = DataSet2Creation(2800)
i, j = DataSet2Creation(2800)

# Final dataset
# Creating a dataframe and saving .csv
Df2= pd.DataFrame({"F1":a, "F2":c, "F3":e, "F4":g, "F5":i,\
    "LabelF1":b, "LabelF2":d, "LabelF3":f, "LabelF4":h, "LabelF5":j})
Df2.to_csv('TimeSeries2.csv')