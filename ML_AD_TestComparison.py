import numpy as np
import pandas as pd 
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support, ConfusionMatrixDisplay)
import joblib
from ML_AD_Preprocessing import stdvector
from ML_AD_Preprocessing import sw_dataset_1
from ML_AD_Preprocessing import sw_dataset_2
import matplotlib.pyplot as plt


#import data and models
TS1 = pd.read_csv("TimeSeries1.csv", index_col = 0)
TS1_Class = pd.read_csv("TimeSeries1_Classification.csv", index_col = 0)
TS1_Class = TS1_Class.iloc[1280:1600, 20:60]
TS1_Class = np.array(TS1_Class).reshape(TS1_Class.size)

TS1_WE_Class = pd.read_csv("TimeSeries1_WE_Classification.csv", index_col = 0)
TS1_WE_Class = np.array(TS1_WE_Class.iloc[1280:1600, 20:60])\
    .reshape(TS1_WE_Class.iloc[1280:1600, 20:60].size)

TS1_Nelson_Class = pd.read_csv("TimeSeries1_Nelson_Classification.csv", index_col = 0)
TS1_Nelson_Class = np.array(TS1_Nelson_Class.iloc[1280:1600, 20:60])\
    .reshape(TS1_Nelson_Class.iloc[1280:1600, 20:60].size)

iforest_1 = joblib.load('ML_iforest_1.sav')
iforest_2 = joblib.load('ML_iforest_2.sav')
svm_1 = joblib.load('ML_svm_1.sav')
svm_2 = joblib.load('ML_svm_2.sav')

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
split= int(4/5 * X_test1.shape[0])

X_test1 = X_test1.iloc[split:X_test1.size, :]
X_test2 = X_test2.iloc[split:X_test2.size, :]

#predict from models
#Predict iForest
predictions_forest1 = iforest_1.predict(np.array(X_test1))
predictions_forest1 = np.array((predictions_forest1 == -1)*1)

predictions_forest2 = iforest_2.predict(np.array(X_test2))
predictions_forest2 = np.array((predictions_forest2 == -1)*1)

#Predict SVM
predictions_SVM1 = svm_1.predict(np.array(X_test1))
predictions_SVM2 = svm_2.predict(np.array(X_test2))

#Confusion Matrices Global
cm_WE = confusion_matrix(TS1_Class, TS1_WE_Class)
cm_Nelson = confusion_matrix(TS1_Class, TS1_Nelson_Class)
cm_forest1 = confusion_matrix(TS1_Class, predictions_forest1)
cm_forest2 = confusion_matrix(TS1_Class, predictions_forest2)
cm_SVM1 = confusion_matrix(TS1_Class, predictions_SVM1)
cm_SVM2 = confusion_matrix(TS1_Class, predictions_SVM2)

#Confusion Matrices per error type
TS1_Class_Normal = TS1_Class[0:1600]
TS1_Class_Cyclic = TS1_Class[1600:3200]
TS1_Class_Systematic = TS1_Class[3200:4800]
TS1_Class_Stratified = TS1_Class[4800:6400]
TS1_Class_ut = TS1_Class[6400:8000]
TS1_Class_dt = TS1_Class[8000:9600]
TS1_Class_us = TS1_Class[9600:11200]
TS1_Class_ds = TS1_Class[11200:12800]

TS1_WE_Class_Normal = TS1_WE_Class[0:1600]
TS1_WE_Class_Cyclic = TS1_WE_Class[1600:3200]
TS1_WE_Class_Systematic = TS1_WE_Class[3200:4800]
TS1_WE_Class_Stratified = TS1_WE_Class[4800:6400]
TS1_WE_Class_ut = TS1_WE_Class[6400:8000]
TS1_WE_Class_dt = TS1_WE_Class[8000:9600]
TS1_WE_Class_us = TS1_WE_Class[9600:11200]
TS1_WE_Class_ds = TS1_WE_Class[11200:12800]

TS1_Nelson_Class_Normal = TS1_Nelson_Class[0:1600]
TS1_Nelson_Class_Cyclic = TS1_Nelson_Class[1600:3200]
TS1_Nelson_Class_Systematic = TS1_Nelson_Class[3200:4800]
TS1_Nelson_Class_Stratified = TS1_Nelson_Class[4800:6400]
TS1_Nelson_Class_ut = TS1_Nelson_Class[6400:8000]
TS1_Nelson_Class_dt = TS1_Nelson_Class[8000:9600]
TS1_Nelson_Class_us = TS1_Nelson_Class[9600:11200]
TS1_Nelson_Class_ds = TS1_Nelson_Class[11200:12800]

predictions_forest1_Normal = predictions_forest1[0:1600]
predictions_forest1_Cyclic = predictions_forest1[1600:3200]
predictions_forest1_Systematic = predictions_forest1[3200:4800]
predictions_forest1_Stratified = predictions_forest1[4800:6400]
predictions_forest1_ut = predictions_forest1[6400:8000]
predictions_forest1_dt = predictions_forest1[8000:9600]
predictions_forest1_us = predictions_forest1[9600:11200]
predictions_forest1_ds = predictions_forest1[11200:12800]

predictions_forest2_Normal = predictions_forest2[0:1600]
predictions_forest2_Cyclic = predictions_forest2[1600:3200]
predictions_forest2_Systematic = predictions_forest2[3200:4800]
predictions_forest2_Stratified = predictions_forest2[4800:6400]
predictions_forest2_ut = predictions_forest2[6400:8000]
predictions_forest2_dt = predictions_forest2[8000:9600]
predictions_forest2_us = predictions_forest2[9600:11200]
predictions_forest2_ds = predictions_forest2[11200:12800]

predictions_SVM1_Normal = predictions_SVM1[0:1600]
predictions_SVM1_Cyclic = predictions_SVM1[1600:3200]
predictions_SVM1_Systematic = predictions_SVM1[3200:4800]
predictions_SVM1_Stratified = predictions_SVM1[4800:6400]
predictions_SVM1_ut = predictions_SVM1[6400:8000]
predictions_SVM1_dt = predictions_SVM1[8000:9600]
predictions_SVM1_us = predictions_SVM1[9600:11200]
predictions_SVM1_ds = predictions_SVM1[11200:12800]

predictions_SVM2_Normal = predictions_SVM2[0:1600]
predictions_SVM2_Cyclic = predictions_SVM2[1600:3200]
predictions_SVM2_Systematic = predictions_SVM2[3200:4800]
predictions_SVM2_Stratified = predictions_SVM2[4800:6400]
predictions_SVM2_ut = predictions_SVM2[6400:8000]
predictions_SVM2_dt = predictions_SVM2[8000:9600]
predictions_SVM2_us = predictions_SVM2[9600:11200]
predictions_SVM2_ds = predictions_SVM2[11200:12800]

cm_WE_normal = confusion_matrix(TS1_Class_Normal, TS1_WE_Class_Normal)
cm_WE_Cyclic = confusion_matrix(TS1_Class_Cyclic, TS1_WE_Class_Cyclic)
cm_WE_Systematic = confusion_matrix(TS1_Class_Systematic, TS1_WE_Class_Systematic)
cm_WE_Stratified = confusion_matrix(TS1_Class_Stratified, TS1_WE_Class_Stratified)
cm_WE_ut = confusion_matrix(TS1_Class_ut, TS1_WE_Class_ut)
cm_WE_dt = confusion_matrix(TS1_Class_dt, TS1_WE_Class_dt)
cm_WE_us = confusion_matrix(TS1_Class_us, TS1_WE_Class_us)
cm_WE_ds = confusion_matrix(TS1_Class_ds, TS1_WE_Class_ds)

cm_Nelson_normal = confusion_matrix(TS1_Class_Normal, TS1_Nelson_Class_Normal)
cm_Nelson_Cyclic = confusion_matrix(TS1_Class_Cyclic, TS1_Nelson_Class_Cyclic)
cm_Nelson_Systematic = confusion_matrix(TS1_Class_Systematic, TS1_Nelson_Class_Systematic)
cm_Nelson_Stratified = confusion_matrix(TS1_Class_Stratified, TS1_Nelson_Class_Stratified)
cm_Nelson_ut = confusion_matrix(TS1_Class_ut, TS1_Nelson_Class_ut)
cm_Nelson_dt = confusion_matrix(TS1_Class_dt, TS1_Nelson_Class_dt)
cm_Nelson_us = confusion_matrix(TS1_Class_us, TS1_Nelson_Class_us)
cm_Nelson_ds = confusion_matrix(TS1_Class_ds, TS1_Nelson_Class_ds)

cm_forest1_normal = confusion_matrix(TS1_Class_Normal, predictions_forest1_Normal)
cm_forest1_Cyclic = confusion_matrix(TS1_Class_Cyclic, predictions_forest1_Cyclic)
cm_forest1_Systematic = confusion_matrix(TS1_Class_Systematic, predictions_forest1_Systematic)
cm_forest1_Stratified = confusion_matrix(TS1_Class_Stratified, predictions_forest1_Stratified)
cm_forest1_ut = confusion_matrix(TS1_Class_ut, predictions_forest1_ut)
cm_forest1_dt = confusion_matrix(TS1_Class_dt, predictions_forest1_dt)
cm_forest1_us = confusion_matrix(TS1_Class_us, predictions_forest1_us)
cm_forest1_ds = confusion_matrix(TS1_Class_ds, predictions_forest1_ds)

cm_forest2_normal = confusion_matrix(TS1_Class_Normal, predictions_forest2_Normal)
cm_forest2_Cyclic = confusion_matrix(TS1_Class_Cyclic, predictions_forest2_Cyclic)
cm_forest2_Systematic = confusion_matrix(TS1_Class_Systematic, predictions_forest2_Systematic)
cm_forest2_Stratified = confusion_matrix(TS1_Class_Stratified, predictions_forest2_Stratified)
cm_forest2_ut = confusion_matrix(TS1_Class_ut, predictions_forest2_ut)
cm_forest2_dt = confusion_matrix(TS1_Class_dt, predictions_forest2_dt)
cm_forest2_us = confusion_matrix(TS1_Class_us, predictions_forest2_us)
cm_forest2_ds = confusion_matrix(TS1_Class_ds, predictions_forest2_ds)

cm_SVM1_normal = confusion_matrix(TS1_Class_Normal, predictions_SVM1_Normal)
cm_SVM1_Cyclic = confusion_matrix(TS1_Class_Cyclic, predictions_SVM1_Cyclic)
cm_SVM1_Systematic = confusion_matrix(TS1_Class_Systematic, predictions_SVM1_Systematic)
cm_SVM1_Stratified = confusion_matrix(TS1_Class_Stratified, predictions_SVM1_Stratified)
cm_SVM1_ut = confusion_matrix(TS1_Class_ut, predictions_SVM1_ut)
cm_SVM1_dt = confusion_matrix(TS1_Class_dt, predictions_SVM1_dt)
cm_SVM1_us = confusion_matrix(TS1_Class_us, predictions_SVM1_us)
cm_SVM1_ds = confusion_matrix(TS1_Class_ds, predictions_SVM1_ds)

cm_SVM2_normal = confusion_matrix(TS1_Class_Normal, predictions_SVM2_Normal)
cm_SVM2_Cyclic = confusion_matrix(TS1_Class_Cyclic, predictions_SVM2_Cyclic)
cm_SVM2_Systematic = confusion_matrix(TS1_Class_Systematic, predictions_SVM2_Systematic)
cm_SVM2_Stratified = confusion_matrix(TS1_Class_Stratified, predictions_SVM2_Stratified)
cm_SVM2_ut = confusion_matrix(TS1_Class_ut, predictions_SVM2_ut)
cm_SVM2_dt = confusion_matrix(TS1_Class_dt, predictions_SVM2_dt)
cm_SVM2_us = confusion_matrix(TS1_Class_us, predictions_SVM2_us)
cm_SVM2_ds = confusion_matrix(TS1_Class_ds, predictions_SVM2_ds)


#Classification Report Global
cf_WE = classification_report(TS1_Class, TS1_WE_Class)
cf_Nelson = classification_report(TS1_Class, TS1_Nelson_Class)
cf_forest1 = classification_report(TS1_Class, predictions_forest1)
cf_forest2 = classification_report(TS1_Class, predictions_forest2)
cf_SVM1 = classification_report(TS1_Class, predictions_SVM1)
cf_SVM2 = classification_report(TS1_Class, predictions_SVM2)

#Classification Report per Error
cr_WE_normal = classification_report(TS1_Class_Normal, TS1_WE_Class_Normal)
cr_WE_Cyclic = classification_report(TS1_Class_Cyclic, TS1_WE_Class_Cyclic)
cr_WE_Systematic = classification_report(TS1_Class_Systematic, TS1_WE_Class_Systematic)
cr_WE_Stratified = classification_report(TS1_Class_Stratified, TS1_WE_Class_Stratified)
cr_WE_ut = classification_report(TS1_Class_ut, TS1_WE_Class_ut)
cr_WE_dt = classification_report(TS1_Class_dt, TS1_WE_Class_dt)
cr_WE_us = classification_report(TS1_Class_us, TS1_WE_Class_us)
cr_WE_ds = classification_report(TS1_Class_ds, TS1_WE_Class_ds)

cr_Nelson_normal = classification_report(TS1_Class_Normal, TS1_Nelson_Class_Normal)
cr_Nelson_Cyclic = classification_report(TS1_Class_Cyclic, TS1_Nelson_Class_Cyclic)
cr_Nelson_Systematic = classification_report(TS1_Class_Systematic, TS1_Nelson_Class_Systematic)
cr_Nelson_Stratified = classification_report(TS1_Class_Stratified, TS1_Nelson_Class_Stratified)
cr_Nelson_ut = classification_report(TS1_Class_ut, TS1_Nelson_Class_ut)
cr_Nelson_dt = classification_report(TS1_Class_dt, TS1_Nelson_Class_dt)
cr_Nelson_us = classification_report(TS1_Class_us, TS1_Nelson_Class_us)
cr_Nelson_ds = classification_report(TS1_Class_ds, TS1_Nelson_Class_ds)

cr_forest1_normal = classification_report(TS1_Class_Normal, predictions_forest1_Normal)
cr_forest1_Cyclic = classification_report(TS1_Class_Cyclic, predictions_forest1_Cyclic)
cr_forest1_Systematic = classification_report(TS1_Class_Systematic, predictions_forest1_Systematic)
cr_forest1_Stratified = classification_report(TS1_Class_Stratified, predictions_forest1_Stratified)
cr_forest1_ut = classification_report(TS1_Class_ut, predictions_forest1_ut)
cr_forest1_dt = classification_report(TS1_Class_dt, predictions_forest1_dt)
cr_forest1_us = classification_report(TS1_Class_us, predictions_forest1_us)
cr_forest1_ds = classification_report(TS1_Class_ds, predictions_forest1_ds)

cr_forest2_normal = classification_report(TS1_Class_Normal, predictions_forest2_Normal)
cr_forest2_Cyclic = classification_report(TS1_Class_Cyclic, predictions_forest2_Cyclic)
cr_forest2_Systematic = classification_report(TS1_Class_Systematic, predictions_forest2_Systematic)
cr_forest2_Stratified = classification_report(TS1_Class_Stratified, predictions_forest2_Stratified)
cr_forest2_ut = classification_report(TS1_Class_ut, predictions_forest2_ut)
cr_forest2_dt = classification_report(TS1_Class_dt, predictions_forest2_dt)
cr_forest2_us = classification_report(TS1_Class_us, predictions_forest2_us)
cr_forest2_ds = classification_report(TS1_Class_ds, predictions_forest2_ds)

cr_SVM1_normal = classification_report(TS1_Class_Normal, predictions_SVM1_Normal)
cr_SVM1_Cyclic = classification_report(TS1_Class_Cyclic, predictions_SVM1_Cyclic)
cr_SVM1_Systematic = classification_report(TS1_Class_Systematic, predictions_SVM1_Systematic)
cr_SVM1_Stratified = classification_report(TS1_Class_Stratified, predictions_SVM1_Stratified)
cr_SVM1_ut = classification_report(TS1_Class_ut, predictions_SVM1_ut)
cr_SVM1_dt = classification_report(TS1_Class_dt, predictions_SVM1_dt)
cr_SVM1_us = classification_report(TS1_Class_us, predictions_SVM1_us)
cr_SVM1_ds = classification_report(TS1_Class_ds, predictions_SVM1_ds)

cr_SVM2_normal = classification_report(TS1_Class_Normal, predictions_SVM2_Normal)
cr_SVM2_Cyclic = classification_report(TS1_Class_Cyclic, predictions_SVM2_Cyclic)
cr_SVM2_Systematic = classification_report(TS1_Class_Systematic, predictions_SVM2_Systematic)
cr_SVM2_Stratified = classification_report(TS1_Class_Stratified, predictions_SVM2_Stratified)
cr_SVM2_ut = classification_report(TS1_Class_ut, predictions_SVM2_ut)
cr_SVM2_dt = classification_report(TS1_Class_dt, predictions_SVM2_dt)
cr_SVM2_us = classification_report(TS1_Class_us, predictions_SVM2_us)
cr_SVM2_ds = classification_report(TS1_Class_ds, predictions_SVM2_ds)