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
X_train = sw_dataset_3(X_train, 20)
X_train = np.transpose(X_train)
scaler = StandardScaler()
scaler = scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_train = pd.DataFrame(X_train,\
     columns = ["last_value", "mean20", "sigma20","mean5", "sigma5", "find_if", "kurtosis","dir_change", 'wavg', 'slope', 'meancross', 'rdist', 'brange'])

#Split dataset for training
split= int(4/5 * X_train.shape[0])

X_train = X_train.iloc[0:split, :]

#Training supervised SVM
#Structuring labels
TS1_Class = pd.read_csv("Data\TimeSeries1_Classification.csv", index_col = 0)
TS1_Class = TS1_Class.iloc[0:1280, 19:60]
X_labels = np.array(TS1_Class).reshape(TS1_Class.size)

#Create model
parameters = {'kernel':['rbf'], 'C':[0.001, 0.01, 0.1, 1, 10, 100],\
              'gamma':['scale', 0.001, 0.01, 0.1, 1, 10, 100]}

SVM = svm.SVC()

#Cross validation and hyperparameter tuning
cv = KFold(n_splits = 5, shuffle=True,  random_state = 123)
sc = make_scorer(f1_score)

SVM_eval = HalvingGridSearchCV(SVM, parameters, cv = cv, verbose = 4, scoring = sc,random_state = 0)
SVM_tuning = SVM_eval.fit(np.array(X_train), X_labels)
SVM = SVM_tuning.best_estimator_
score_svm3 = SVM_tuning.cv_results_

# save the model and tuning to to PC

joblib.dump(SVM, 'ML_Models\ML_SVM.sav')

joblib.dump(SVM_tuning, 'ML_Models\ML_SVM.pkl')