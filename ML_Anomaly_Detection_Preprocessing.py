import numpy as np
from sklearn.preprocessing import StandardScaler

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

def sw_dataset_1(X, time_steps):
    mu, sigma = [], []
    for l in range (X.shape[0]):
        for i in range(X.shape[1] - time_steps):
            v = X[l, i:(i + time_steps)]
            m = np.mean(v)
            s = np.std(v)
            mu.append(m)
            sigma.append(s)
    return np.array([mu, sigma])

def sw_dataset_2(X, time_steps):
    mu, sigma = [], []
    for l in range (X.shape[0]):
        for i in range(X.shape[1] - time_steps):
            v = X[l, i:(i + time_steps)]
            m = np.mean(v)
            s = np.std(v)
            mu.append(m)
            sigma.append(s)
    return np.array([mu, sigma])