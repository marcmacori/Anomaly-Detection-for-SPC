#libraries
import numpy as np
import pandas as pd
import SPC_AD_library as sp

#import data
TS = pd.read_csv('Data\TimeSeries1.csv', index_col = 0)

#automate data implementation
def AD_in_data(data, numTS, numnorm, rules, time_steps):
    c = []
    data = np.array(data)

    for i in range(numTS):
        
        a = []

        X, R, MR = sp.analyze(data[i, 0:numnorm])

        CL = X[0]
        STD = X[1]
        LCLr = R[0]
        UCLr = R[1]

        for l in range(data.shape[1] - time_steps+1):
            
            v = data[i, l:(l + time_steps)]
            b = sp.apply_rules(v, sp.analyze(v)[2], CL, STD, LCLr, UCLr, rules)
            if any(b==1):
                a.append(1)
            else:
                a.append(0)
        
        c.append(a)

    c=np.array(c)

    return(c)

anomalies = AD_in_data(TS, len(TS), 20, "Nelson", 20)
anomalies = pd.DataFrame(anomalies)
anomalies.to_csv("Data\TimeSeries1_Nelson_Classification.csv")

anomalies = AD_in_data(TS, len(TS), 20, "WE",20)
anomalies = pd.DataFrame(anomalies)
anomalies.to_csv("Data\TimeSeries1_WE_Classification.csv")