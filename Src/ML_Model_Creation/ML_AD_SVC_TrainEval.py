#Libraries
import numpy as np
import pandas as pd 
from sklearnex import patch_sklearn 
patch_sklearn()
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, KFold 
from sklearn.metrics import f1_score, make_scorer, accuracy_score, precision_score, recall_score, average_precision_score
import joblib
import sys
sys.path.append('C:/Users/Marc/Desktop/TFG/R Files/Anomaly Detection for SPC')
from Src.FeatureExt.ML_AD_Preprocessing import *

#Pre-processing
#Import data
TS1 = pd.read_csv("Data\TimeSeries1.csv", index_col = 0)

#Standardize data based on first 20 points of chart, which is supposed in control       
X_train = stdvector(TS1)  

#Creating sliding windows for each TS and get features
X_train1 = sw_dataset_1(X_train, 20)
X_train1 = pd.DataFrame(np.transpose(X_train1))
scaler = StandardScaler()
scaler = scaler.fit(X_train1)
X_train1 = scaler.transform(X_train1)
X_train1 = pd.DataFrame(X_train1, columns = ["mean20", "sigma20"])

X_train2 = sw_dataset_2(X_train, 20)
X_train2 = pd.DataFrame(np.transpose(X_train2))
scaler = StandardScaler()
scaler = scaler.fit(X_train2)
X_train2 = scaler.transform(X_train2)
X_train2 = pd.DataFrame(X_train2,\
     columns = ["last_value", "mean20", "sigma20","mean5", "sigma5", "find_if", "kurtosis"])

X_train3 = sw_dataset_3(X_train, 20)
X_train3 = np.transpose(X_train3)
scaler = StandardScaler()
scaler = scaler.fit(X_train3)
X_train3 = scaler.transform(X_train3)
X_train3 = pd.DataFrame(X_train3,\
     columns = ["last_value", "mean20", "sigma20","mean5", "sigma5", "find_if", "kurtosis","dir_change", 'wavg', 'slope', 'meancross', 'rdist', 'brange'])

#Split dataset for training
split= int(4/5 * X_train1.shape[0])

X_train1 = X_train1.iloc[0:split, :]
X_train2 = X_train2.iloc[0:split, :]
X_train3 = X_train3.iloc[0:split, :]

#Training supervised SVM
#Structuring labels
TS1_Class = pd.read_csv("Data\TimeSeries1_Classification.csv", index_col = 0)
TS1_Class = TS1_Class.iloc[0:1280, 19:60]
X_labels = np.array(TS1_Class).reshape(TS1_Class.size)

#Create model
parameters = {'kernel':['rbf'], 'C':[0.001, 0.01, 0.1, 1, 10, 100],\
              'gamma':['scale', 0.001, 0.01, 0.1, 1, 10, 100]}
svm_1 = svm.SVC() # Gaussian Radial Basis Function Kernel
svm_2 = svm.SVC() 
svm_3 = svm.SVC()

#Cross validation and hyperparameter tuning
cv = KFold(n_splits = 5, shuffle=True,  random_state = 123)
sc = make_scorer(f1_score)

svm_1_eval = HalvingGridSearchCV(svm_1, parameters, cv = cv, verbose = 4, scoring = sc,random_state = 0)
svm_1_tuning = svm_1_eval.fit(np.array(X_train1), X_labels)
svm_1 = svm_1_tuning.best_estimator_
score_svm1 = svm_1_tuning.cv_results_

svm_2_eval = HalvingGridSearchCV(svm_2, parameters, cv = cv, verbose = 4, scoring = sc,random_state = 0)
svm_2_tuning = svm_2_eval.fit(np.array(X_train2), X_labels)
svm_2 = svm_2_tuning.best_estimator_
score_svm2 = svm_2_tuning.cv_results_

svm_3_eval = HalvingGridSearchCV(svm_3, parameters, cv = cv, verbose = 4, scoring = sc,random_state = 0)
svm_3_tuning = svm_3_eval.fit(np.array(X_train3), X_labels)
svm_3 = svm_3_tuning.best_estimator_
score_svm3 = svm_3_tuning.cv_results_

# save the model and tuning to to PC
joblib.dump(svm_1, 'ML_Models\ML_svm_1.sav')
joblib.dump(svm_2, 'ML_Models\\ML_svm_2.sav')
joblib.dump(svm_3, 'ML_Models\ML_svm_3.sav')

joblib.dump(svm_1_tuning, 'ML_Models\ML_svm_1_tuning.pkl')
joblib.dump(svm_2_tuning, 'ML_Models\\ML_svm_2_tuning.pkl')
joblib.dump(svm_3_tuning, 'ML_Models\ML_svm_3.pkl')