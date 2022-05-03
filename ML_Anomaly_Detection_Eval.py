#Libraries
import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix
import joblib
from ML_Anomaly_Detection_Preprocessing import stdvector
from ML_Anomaly_Detection_Preprocessing import sw_dataset

#import data and models
TS1 = pd.read_csv("TimeSeries1.csv", index_col = 0)
TS1_Class = pd.read_csv("TimeSeries1_Classification.csv", index_col = 0)
TS1_Class = np.array(TS1_Class.iloc[:, 20:60]).reshape(TS1_Class.iloc[:, 20:60].size)

TS1_WE_Class = pd.read_csv("TimeSeries1_WE_Classification.csv", index_col = 0)
TS1_WE_Class = np.array(TS1_WE_Class.iloc[20:60,:]).reshape(TS1_WE_Class.iloc[20:60,:].size)
TS1_WE_Class = np.transpose(TS1_WE_Class)

iforest_20 = joblib.load('iforest_20.sav')
svm_20 = joblib.load('svm_20.sav')

#data preprocessing
X_train = TS1
X_train = stdvector(X_train)  
X_train = sw_dataset(X_train, 20)
X_train = pd.DataFrame(np.transpose(X_train), columns = ["mean", "sigma"])

#predict from models
#Predict iForest
predictions_forest = iforest_20.predict(X_train)
predictions_forest = np.array((predictions_forest == -1)*1)

#Predict SVM
predictions_SVM = svm_20.predict(X_train)

#Confusion Matrices
cm_iforest = confusion_matrix(TS1_Class, predictions_forest)
cm_SVM = confusion_matrix(TS1_Class, predictions_SVM)
cm_WE = confusion_matrix(TS1_Class, TS1_WE_Class)