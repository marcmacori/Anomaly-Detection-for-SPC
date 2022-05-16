#Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.ensemble import IsolationForest
import joblib
from ML_AD_Preprocessing import stdvector
from ML_AD_Preprocessing import sw_dataset_1
from ML_AD_Preprocessing import sw_dataset_2

#Pre-processing
#Import data
TS1 = pd.read_csv("TimeSeries1.csv", index_col = 0)

#Standardize data based on first 20 points of chart, which is supposed in control       
X_train = stdvector(TS1)  

#Creating sliding windows for each TS and get features
X_train1 = sw_dataset_1(X_train, 20)
X_train1 = pd.DataFrame(np.transpose(X_train1), columns = ["mean", "sigma"])

X_train2 = sw_dataset_2(X_train, 20)
X_train2 = pd.DataFrame(np.transpose(X_train2),\
     columns = ["last_value", "mean", "sigma","mean5", "sigma5", "findif", "kurtosis"])

#Split dataset for training
split= int(4/5 * X_train1.shape[0])

X_train1 = X_train1.iloc[0:split, :]
X_train2 = X_train2.iloc[0:split, :]

#Training isolation forest
#define contamination
TS1_Class = pd.read_csv("TimeSeries1_Classification.csv", index_col = 0)
TS1_Class = TS1_Class.iloc[0:1280, 20:60]
contam = TS1_Class.values.sum() / TS1_Class.size

#Define and employ algorithm on dataset
random_state = np.random.RandomState(42)
iforest_1 = IsolationForest(n_estimators=100, max_samples = 'auto',\
    contamination = float(contam), random_state = random_state)
random_state = np.random.RandomState(1234)
iforest_2 = IsolationForest(n_estimators=100, max_samples = 'auto',\
    contamination = float(contam), random_state = random_state)
iforest_1.fit(np.array(X_train1))
iforest_2.fit(np.array(X_train2))

# save the model to disk
joblib.dump(iforest_1, 'ML_iforest_1.sav')
joblib.dump(iforest_2, 'ML_iforest_2.sav')