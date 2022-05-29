#libraries
import numpy as np
import pandas as pd
import SPC_AD_library as sp

#import data
TS = pd.read_csv('Data\TimeSeries1.csv', index_col = 0)

#automate data implementation
def AD_in_data(data, numTS, numnorm, rules):
    a = []
    data = np.array(data)

    for i in range(numTS):
        
        X, R, MR = sp.analyze(data[i, 0:numnorm])

        CL = X[0]
        STD = X[1]
        LCLr = R[0]
        UCLr = R[1]

        b = sp.apply_rules(data[i, :], sp.analyze(data[i, :])[2], CL, STD, LCLr, UCLr, rules)

        a.append(b)

    return(a)

anomalies = AD_in_data(TS, len(TS), 20, "Nelson")
anomalies = pd.DataFrame(anomalies)
anomalies.to_csv("Data\TimeSeries1_Nelson_Classification.csv")

anomalies = AD_in_data(TS, len(TS), 20, "WE")
anomalies = pd.DataFrame(anomalies)
anomalies.to_csv("Data\TimeSeries1_WE_Classification.csv")