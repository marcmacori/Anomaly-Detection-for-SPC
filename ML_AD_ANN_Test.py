import numpy as np
import pandas as pd 
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)
import joblib
from ML_AD_Preprocessing import stdvector
from ML_AD_Preprocessing import sw_dataset_1
from ML_AD_Preprocessing import sw_dataset_2
from keras.models import load_model
import matplotlib.pyplot as plt


#import data and models
TS1 = pd.read_csv("TimeSeries1.csv", index_col = 0)
TS1_Class = pd.read_csv("TimeSeries1_Classification.csv", index_col = 0)
TS1_Class = TS1_Class.iloc[960:1600, 20:60]
TS1_Class = np.array(TS1_Class).reshape(TS1_Class.size)

TS1_WE_Class = pd.read_csv("TimeSeries1_WE_Classification.csv", index_col = 0)
TS1_WE_Class = np.array(TS1_WE_Class.iloc[960:1600, 20:60])\
    .reshape(TS1_WE_Class.iloc[960:1600, 20:60].size)

TS1_Nelson_Class = pd.read_csv("TimeSeries1_Nelson_Classification.csv", index_col = 0)
TS1_Nelson_Class = np.array(TS1_Nelson_Class.iloc[960:1600, 20:60])\
    .reshape(TS1_Nelson_Class.iloc[960:1600, 20:60].size)

ANN_2 = load_model('ML_ANN_2.h5')

#Pre-processing
#Standardize data based on first 20 points of chart, which is supposed in control       
X_test = stdvector(TS1)  

#Creating sliding windows for each TS and get features
X_test1 = sw_dataset_1(X_test, 20)
X_test1 = pd.DataFrame(np.transpose(X_test1), columns = ["mean", "sigma"])

X_test2 = sw_dataset_2(X_test, 20)
X_test2 = pd.DataFrame(np.transpose(X_test2),\
     columns = ["last_value", "mean", "sigma","mean5", "sigma5", "findif", "kurtosis"])

#Split dataset for testing
split= int(3/5 * X_test1.shape[0])

X_test1 = X_test1.iloc[split:X_test1.size, :]
X_test2 = X_test2.iloc[split:X_test2.size, :]

#Predict ANN
predictions_ANN2 = ANN_2.predict(X_test2)
rounded = [int(round(x[0])) for x in predictions_ANN2]


#Confusion Matrices
cm_WE = confusion_matrix(TS1_Class, TS1_WE_Class)
cm_Nelson = confusion_matrix(TS1_Class, TS1_Nelson_Class)
cm_ANN2 = confusion_matrix(TS1_Class, rounded)