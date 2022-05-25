# Synthetic Data Creator in Python
import numpy as np
import pandas as pd 
from DataCreation import *
# Creating Dataset number 2. A multifeature dataset, with 90000 points and anomalies
# starting from point 45000. Each point is labeled as an anomaly or not in a 
# correspoinding vector with the same logic as above
def DataSet2Creation(len):
    
    mu = np.random.uniform(1, 50, size=1)
    sg = np.random.uniform(mu/20, mu/5, size=1)
    zin = np.zeros(2000)

    a = [normal(mu, sg, 2000), zin]

    Xs = a[0]
    Ys = a[1]

    while (Ys.shape[0] < len):
        
        lonn = np.random.randint(200, 500)
        lono = np.random.randint(200, 500)
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

a, b = DataSet2Creation(10000)
c, d = DataSet2Creation(10000)
e, f = DataSet2Creation(10000)
g, h = DataSet2Creation(10000)
i, j = DataSet2Creation(10000)

# Final dataset
# Creating a dataframe and saving .csv
Df2= pd.DataFrame({"F1":a, "F2":c, "F3":e, "F4":g, "F5":i,\
    "LabelF1":b, "LabelF2":d, "LabelF3":f, "LabelF4":h, "LabelF5":j})
Df2.to_csv('Data\TimeSeries2.csv')