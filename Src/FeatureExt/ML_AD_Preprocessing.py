import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis
from scipy.stats import linregress

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
        for i in range(X.shape[1] - time_steps+1):
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
        for i in range(X.shape[1] - time_steps+1):
            v = X[l, i:(i + time_steps)]
            lval = v[time_steps-1]
            m = np.mean(v)
            s = np.std(v)
            m5 = np.mean(v[time_steps-6:time_steps])
            s5 = np.std(v[time_steps-6:time_steps])
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
    mean20, sigma20, lastval, mean5, sigma5, diffin, kurt, dirchange, wavg, slope, meancross, rdist, brange =\
         [], [], [], [], [], [], [],[],[],[],[],[],[]
    for l in range (X.shape[0]):

        for i in range(X.shape[1] - time_steps+1):
            t=np.arange(time_steps)
            v = X[l, i:(i + time_steps)]
            lval = v[time_steps-1]
            m = np.mean(v)
            s = np.std(v)
            m5 = np.mean(v[time_steps-6:time_steps])
            s5 = np.std(v[time_steps-6:time_steps])
            dif = (v[time_steps - 1] - v[time_steps - 2])/2
            kt = kurtosis(v)
            sl=linregress(t,v).slope
            weighavg=np.sum(np.multiply(t, v))/time_steps
            
            directionchange = 0
            for k in range(len(v)-2):
                if np.sign(v[k+1] - v[k]) != np.sign(v[k+2] - v[k+1]):
                    directionchange = directionchange + 1

            meancr = 0
            for k in range(len(v)-1):
                if ((v[k+1]<m and v[k]>m) or (v[k+1]>m and v[k]<m)):
                    meancr = meancr + 1

            rd = []
            for k in range(len(v)-1):
                val=np.sqrt((t[k+1]-t[k])**2+(v[k+1]-v[k])**2)
                rd.append(val)


            rd=(np.sum(rd)/(len(v)-1))/s

            sect1=v[0:np.trunc((1/4*len(v))).astype(int)]
            sect2=v[np.trunc((1/4*len(v))).astype(int):np.trunc((2/4*len(v))).astype(int)]
            sect3=v[np.trunc((2/4*len(v))).astype(int):np.trunc((3/4*len(v))).astype(int)]
            sect4=v[np.trunc((3/4*len(v))).astype(int):np.trunc((4/4*len(v))).astype(int)]


            slb=[linregress(t[0:(len(sect1))*2],np.concatenate((sect1, sect2))).slope]
            slb.append(linregress(t[0:(len(sect1))*2],np.concatenate((sect1, sect3))).slope)
            slb.append(linregress(t[0:(len(sect1))*2],np.concatenate((sect1, sect4))).slope)
            slb.append(linregress(t[0:(len(sect1))*2],np.concatenate((sect2, sect3))).slope)
            slb.append(linregress(t[0:(len(sect1))*2],np.concatenate((sect2, sect4))).slope)
            slb.append(linregress(t[0:(len(sect1))*2],np.concatenate((sect3, sect4))).slope)
            slb=np.array(slb)
            bran=np.max(slb)-np.min(slb)

            lastval.append(lval)
            mean20.append(m)
            sigma20.append(s)
            mean5.append(m5)
            sigma5.append(s5)
            diffin.append(dif)
            kurt.append(kt)
            dirchange.append(directionchange)
            wavg.append(weighavg)
            slope.append(sl)
            meancross.append(meancr)
            rdist.append(rd)
            brange.append(bran)

    return np.array([lastval, mean20, sigma20, mean5, sigma5, diffin, kurt, dirchange, wavg, slope, meancross, rdist, brange])