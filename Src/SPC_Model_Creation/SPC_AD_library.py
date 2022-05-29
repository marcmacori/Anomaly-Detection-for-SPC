#Libraries
import numpy as np
import matplotlib.pyplot as plt

#Calculate control Limits for a Shewart Individual Control Chart
def analyze(data):
    E2 = 2.66
    D3 = 0
    D4 = 3.266
    len = np.size(data)
    mean = np.mean(data)
    std_dev = np.std(data)
    
    moving_range = []

    for i in range(len-1):
        mr = np.abs(data[i+1] - data[i])
        moving_range.append(mr)

    R = np.mean(moving_range)

    LCLr = D3*R
    UCLr = D4*R 
    CLr = R

    CLx = mean
    STD = R*E2

    return[np.array([CLx, STD]), np.array([LCLr,UCLr,CLr]), np.array(moving_range)]

#Create sliding windows for rule application
def sliding_window(X, window_len):
    chunk = []

    for i in range(len(X) - window_len + 1):
        v = X[i:(i + window_len)]
        chunk.append(v)

    return np.array(chunk)

#Create rules functions
def rule1(data, LCL, UCL):
    """One point is more than 3 standard deviations from the mean."""
    results = np.zeros(len(data))
    for i in range(len(data)):
        if data[i] < LCL:
            results[i] = 1
        elif data[i] > UCL:
            results[i] = 1

    return results.astype(int)

def rule2(data, CL, STD):
    """Nine (or more) points in a row are on the same side of the mean."""
    results = np.zeros(len(data))
    chunks = sliding_window(data, 9)

    for i in range(len(chunks)):
        if np.sum(chunks[i]<CL) >= 9 or np.sum(chunks[i]>CL) >=9:
            results[i:i+9] = 1
    return results.astype(int)

def rule3(data, CL, STD):
    """Six (or more) points in a row are continually increasing (or decreasing)."""
    results = np.zeros(len(data))
    chunks = sliding_window(data, 6)
    
    for i in range(len(chunks)):
        numinc, numdec= 0, 0

        if chunks[i][0]<chunks[i][1]:
            for k in range(len(chunks[i])-1):

                if chunks[i][k]<chunks[i][k+1]:
                    numinc = numinc + 1
        else:
            for k in range(len(chunks[i])-1):

                if chunks[i][k]<chunks[i][k+1]:
                    numdec =  numdec + 1

        if numinc>= 5 or numdec>=5:
            results[i:i+6] = 1

    return results.astype(int)


def rule4(data, CL, STD):
    """Fourteen (or more) points in a row alternate in direction, increasing then decreasing."""
    results = np.zeros(len(data))
    chunks = sliding_window(data, 14)

    for i in range(len(chunks)):
        directionchange = 0

        for k in range(len(chunks[i])-2):

            if np.sign(chunks[i][k+1] - chunks[i][k]) != np.sign(chunks[i][k+2] - chunks[i][k+1]):
                directionchange = directionchange + 1

        if np.sum(directionchange)>= 12:
            results[i:i+14] = 1

    return results.astype(int)


def rule5(data, CL, STD):
    """Two (or three) out of three points in a row are more than 2 standard 
    deviations from the mean in the same direction."""
    results = np.zeros(len(data))
    chunks = sliding_window(data, 3)

    for i in range(len(chunks)):
        if np.sum(chunks[i]<(CL - 2/3*STD)) >= 2:
            results[i:i+3] = chunks[i] < (CL - 2/3*STD)
        elif np.sum(chunks[i]>(CL + 2/3*STD)) >= 2:
            results[i:i+3] = chunks[i] > CL + 2/3*STD

    return results.astype(int)

def rule6(data, CL, STD):
    """Four (or five) out of five points in a row are more than 1 standard 
    deviation from the mean in the same direction."""

    results = np.zeros(len(data))
    chunks = sliding_window(data, 5)

    for i in range(len(chunks)):
        if np.sum(chunks[i]<(CL - 1/3*STD)) >= 4:
            results[i:i+5] = chunks[i] < (CL - 1/3*STD)
        elif np.sum(chunks[i]>(CL + 1/3*STD)) >= 4:
            results[i:i+5] = chunks[i] > CL + 1/3*STD

    return results.astype(int)

def rule7(data, CL, STD):
    """Fifteen points in a row are all within 1 standard 
    deviation of the mean on either side of the mean."""
    results = np.zeros(len(data))
    chunks = sliding_window(data, 15)

    for i in range(len(chunks)):
        if all((CL - 1/3*STD) < i < (CL + 1/3*STD) for i in chunks[i]):
            results[i:i+15] = 1

    return results.astype(int)

def rule8(data, CL, STD):
    """Eight points in a row exist, but none within 1 standard deviation
    of the mean, and the points are in both directions from the mean."""

    results = np.zeros(len(data))
    chunks = sliding_window(data, 8)

    for i in range(len(chunks)):
        if all(i < (CL - 1/3*STD) or i > (CL + 1/3*STD) for i in chunks[i])\
                and any(i < (CL - 1/3*STD) for i in chunks[i])\
                and any(i > (CL + 1/3*STD) for i in chunks[i]):
                results[i:i+8] = 1

    return results.astype(int)

#Apply all rules to a control chart
def apply_rules(data, MR, CL, STD, LCLr, UCLr, rules):

    if rules == 'Nelson':
        anomaly = np.zeros(len(data))
        rules = [rule2, rule3, rule4, rule5, rule6, rule7, rule8]
        anomaly = rule1(data, CL - STD, CL + STD)
        anomaly = anomaly + np.append(0,rule1(MR, LCLr, UCLr))

        for i in range(len(rules)):
            v = rules[i](data, CL, STD)
            anomaly = anomaly + v

    if rules == 'WE':
        rules = [rule2, rule5, rule6]
        anomaly = rule1(data, CL - STD, CL + STD)
        anomaly = anomaly + np.append(0,rule1(MR, LCLr, UCLr))
        
        for i in range(len(rules)):
            v = rules[i](data, CL, STD)
            anomaly = anomaly + v

    return np.array(anomaly>=1).astype(int)

#Plot a control chart
def plot_chart(data, anomalies, LCL, UCL, CL):

    fig, axs = plt.subplots()
    axs.scatter(np.arange(0, len(data)), data, marker='o', c=anomalies, cmap='bwr')
    axs.plot(np.arange(0, len(data)), data, linestyle = '-', color = "black")
    axs.axhline(CL, color='blue')
    axs.axhline(LCL, color = 'red', linestyle = '-')
    axs.axhline(UCL, color = 'red', linestyle = '-')
    axs.axhline(CL+1/3*(UCL- CL), color = 'red', linestyle = 'dashed', alpha=0.2)
    axs.axhline(CL+1/3*(LCL- CL), color = 'red', linestyle = 'dashed', alpha=0.2)
    axs.axhline(CL+2/3*(UCL- CL), color = 'red', linestyle = 'dashed', alpha=0.4)
    axs.axhline(CL+2/3*(LCL- CL), color = 'red', linestyle = 'dashed', alpha=0.4)
    axs.set_title('Individual Chart')
    axs.set(xlabel='Observation Number', ylabel='QC')