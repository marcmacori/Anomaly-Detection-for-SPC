# Synthetic Data Creator in Python
import numpy as np
import pandas as pd 
from DataCreation import *

# creation of standard dataset: 40 time series with no anomalies and 
# 280 time series with anomalies starting from point 20 and 
# starting with a 1 position delay for each new time series up until 60. 
# This is repeated for each type of anomaly. All points are labeled as in a corresponding matrix
# O corresponding to non-anomaly and 1 correspoding to anomaly
def DataSetCreation(mu, sigma, numobs):
    Xs, Ys = [], [],
    for i in range(40):
        X = normal(mu, sigma, numobs)
        Y = np.zeros(numobs)
        Xs.append(X)
        Ys.append(Y)
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
    return np.array(Xs), np.array(Ys).astype(int) 

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
Df1.to_csv('Data\TimeSeries1.csv')
Df1_Classification.to_csv('Data\TimeSeries1_Classification.csv')


#Creating a normal dataset for semi-supervised training
def NormalDataSetCreation(mu, sigma, numobs):
    Xs, Ys = [], [],
    for i in range(1000):
        X = normal(mu, sigma, numobs)
        Y = np.zeros(numobs)
        Xs.append(X)
        Ys.append(Y)
    return np.array(Xs), np.array(Ys) 

TS1, TS_Classification1 = NormalDataSetCreation(10, 1, 60)

# Creating a dataframe and saving .csv
Df1= pd.DataFrame(TS1)
Df1.to_csv('Data\TimeSeriesNormal.csv')