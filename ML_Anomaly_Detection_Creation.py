#Libraries
import numpy as np
import pandas as pd 
from sklearn.ensemble import IsolationForest
from sklearn import svm
import joblib
from ML_Anomaly_Detection_Preprocessing import stdvector
from ML_Anomaly_Detection_Preprocessing import sw_dataset_1

#Pre-processing
#Import and split data
TS1 = pd.read_csv("TimeSeries1.csv", index_col = 0)
X_train = TS1

#Standardize data based on first 20 points of chart, which is supposed in control       
X_train = stdvector(X_train)  

#Creating sliding windows for each TS and finding their mean and sigma
X_train = sw_dataset_1(X_train, 20)
X_train = pd.DataFrame(np.transpose(X_train), columns = ["mean", "sigma"])
TS1_Class = pd.read_csv("TimeSeries1_Classification.csv", index_col = 0)
TS1_Class = TS1_Class.iloc[:, 20:60]

#Training isolation forest for both features
#define contamination
contam = TS1_Class.values.sum() / TS1_Class.size

random_state = np.random.RandomState(42)
iforest_20 = IsolationForest(n_estimators=100, max_samples = 'auto',\
    contamination = float(contam), random_state = random_state)

iforest_20.fit(X_train)

#Training supervised SVM as trial
#Structuring labels
X_labels = np.array(TS1_Class).reshape(TS1_Class.size)

#Create model
svm_20 = svm.SVC(kernel='rbf') # Gaussian Radial Basis Function Kernel

#Train the model using the training sets
svm_20.fit(X_train, X_labels)

# save the model to disk
filename = 'svm_20.sav'
joblib.dump(svm_20, filename)

# save the model to disk
filename = 'iforest_20.sav'
joblib.dump(iforest_20, filename)