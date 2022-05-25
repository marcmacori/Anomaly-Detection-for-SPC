import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis

#########################################################################
#Normalize Data
#########################################################################
def stdvector(data):
    data = np.array(data)
    a = []
    for i in range(len(data)):
        v = data[i, 0:20]
        scaler = StandardScaler()
        scaler = scaler.fit(v.reshape(-1,1))
        scaled = scaler.transform(data[i, :].reshape(-1,1))
        a.append(scaled)
    return(np.array(a).reshape(np.shape(data)))
#########################################################################
#Sliding window feature extractions trial 1
#########################################################################
def sw_dataset_1(X, time_steps):
    mean20, sigma20 = [], []
    for l in range (X.shape[0]):
        for i in range(X.shape[1] - time_steps):
            v = X[l, i:(i + time_steps)]
            m = np.mean(v)
            s = np.std(v)
            mean20.append(m)
            sigma20.append(s)
    return np.array([mean20, sigma20])
#########################################################################
#Sliding window feature extractions trial 2
#########################################################################
def sw_dataset_2(X, time_steps):
    mean20, sigma20, lastval, mean5, sigma5, diffin, kurt = [], [], [], [], [], [], []
    for l in range (X.shape[0]):
        for i in range(X.shape[1] - time_steps):
            v = X[l, i:(i + time_steps)]
            lval = v[time_steps-1]
            m = np.mean(v)
            s = np.std(v)
            m5 = np.mean(v[time_steps-5:time_steps])
            s5 = np.std(v[time_steps-5:time_steps])
            dif = (v[time_steps - 1] - v[time_steps - 2])/2
            kt = kurtosis(v)
            lastval.append(lval)
            mean20.append(m)
            sigma20.append(s)
            mean5.append(m5)
            sigma5.append(s5)
            diffin.append(dif)
            kurt.append(kt)
    return np.array([lastval, mean20, sigma20, mean5, sigma5, diffin, kurt])
#########################################################################
#Sliding window feature extractions trial 3
#########################################################################
def sw_dataset_3(X, time_steps):
    mean20, sigma20, lastval, mean5, sigma5, diffin, kurt, dirchange =\
         [], [], [], [], [], [], [],[]
    for l in range (X.shape[0]):

        for i in range(X.shape[1] - time_steps):
            v = X[l, i:(i + time_steps)]
            lval = v[time_steps-1]
            m = np.mean(v)
            s = np.std(v)
            m5 = np.mean(v[time_steps-5:time_steps])
            s5 = np.std(v[time_steps-5:time_steps])
            dif = (v[time_steps - 1] - v[time_steps - 2])/2
            kt = kurtosis(v)

            directionchange = 0
            for k in range(len(v)-2):
                if np.sign(v[k+1] - v[k]) != np.sign(v[k+2] - v[k+1]):
                    directionchange = directionchange + 1

            lastval.append(lval)
            mean20.append(m)
            sigma20.append(s)
            mean5.append(m5)
            sigma5.append(s5)
            diffin.append(dif)
            kurt.append(kt)
            dirchange.append(directionchange)

    return np.array([lastval, mean20, sigma20, mean5, sigma5, diffin, kurt, dirchange])
#########################################################################
#Sliding window feature extractions trial 4
#########################################################################
def sw_dataset_4(X, time_steps):
    V = []
    for l in range (X.shape[0]):
        for i in range(X.shape[1] - time_steps):
            v = X[l, i:(i + time_steps)]
            V.append(v)
    return np.array(V)