#Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn import svm
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, KFold, ShuffleSplit 
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
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

#Training supervised SVM
#Structuring labels
#define contamination
TS1_Class = pd.read_csv("TimeSeries1_Classification.csv", index_col = 0)
TS1_Class = TS1_Class.iloc[0:1280, 20:60]
X_labels = np.array(TS1_Class).reshape(TS1_Class.size)

#Create model
parameters = {'kernel':['rbf'], 'C':[0.01, 1, 10], 'gamma':[0.001, 0.01, 1]}
svm_1 = svm.SVC() # Gaussian Radial Basis Function Kernel
svm_2 = svm.SVC() # Gaussian Radial Basis Function Kernel

#Cross validation and hyperparameter tuning
cv = KFold(n_splits = 5, shuffle=True,  random_state = 123)

svm_1_eval = HalvingGridSearchCV(svm_1, parameters, cv = cv, verbose = 4, random_state=0)
svm_1_tuning = svm_1_eval.fit(np.array(X_train1), X_labels)
svm_1 = svm_1_tuning.best_estimator_
score_svm1 = svm_1_tuning.cv_results_

svm_2_eval = HalvingGridSearchCV(svm_2, parameters, cv = cv, verbose = 4, random_state=0)
svm_2_tuning = svm_2_eval.fit(np.array(X_train2), X_labels)
svm_2 = svm_2_tuning.best_estimator_
score_svm2 = svm_2_tuning.cv_results_

# save the model to disk
joblib.dump(svm_1, 'ML_svm_1.sav')
joblib.dump(svm_2, 'ML_svm_2.sav')