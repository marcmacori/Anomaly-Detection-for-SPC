#Libraries
import numpy as np
import pandas as pd 
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import RandomizedSearchCV, KFold
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
X_train1 = pd.DataFrame(np.transpose(X_train1), columns = ["mean20", "sigma20"])

X_train2 = sw_dataset_2(X_train, 20)
X_train2 = pd.DataFrame(np.transpose(X_train2),\
     columns = ["last_value", "mean20", "sigma20","mean5", "sigma5", "find_if", "kurtosis"])

X_train3 = sw_dataset_3(X_train, 20)
X_train3 = np.transpose(X_train3)
X_train3 = pd.DataFrame(X_train3,\
     columns = ["last_value", "mean20", "sigma20","mean5", "sigma5", "find_if", "kurtosis","dir_change", 'wavg', 'slope', 'meancross', 'rdist', 'brange'])

#Split dataset for training
split= int(4/5 * X_train1.shape[0])

X_train1 = X_train1.iloc[0:split, :]
X_train2 = X_train2.iloc[0:split, :]
X_train3 = X_train3.iloc[0:split, :]

#Structuring labels for hyperparameter tuning
TS1_Class = pd.read_csv("Data\TimeSeries1_Classification.csv", index_col = 0)
TS1_Class = TS1_Class.iloc[0:1280, 19:60]
X_labels = np.array(TS1_Class).reshape(TS1_Class.size)
X_labels[X_labels == 1] = -1
X_labels[X_labels == 0] = 1


#Define Model
random_state = np.random.RandomState(42)
iforest_1 = IsolationForest()
iforest_2 = IsolationForest()
iforest_3 = IsolationForest()

#Cross validation and hyperparameter tuning
# Method of selecting samples for training each tree
bootstrap = [True, False]
#Contamination 
contamination = ["auto", 0.1, 0.2, 0.3, 0.4, 0.5]
# Number of features to consider at every split
max_features = [0.25, 0.5, 1.0]
# Number of features to consider at every split
max_samples = ["auto", 32, 64, 128, 512, 1024]
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
#Number of Jobs
n_jobs = [1, 5, 10, 20, 30]

# Create the random grid
grid = {'bootstrap': bootstrap,
                'contamination': contamination,
                'max_features': max_features,
                'max_samples': max_samples,
                'n_estimators': n_estimators,
                'random_state': [0,42],
                'warm_start':[False, True],
                'n_jobs': n_jobs}

cv = KFold(n_splits = 5, shuffle=True,  random_state = 123)
sc = {"PR_AUC_score": make_scorer(average_precision_score), "Accuracy": make_scorer(accuracy_score), "Precision": make_scorer(precision_score),\
     "Recall": make_scorer(recall_score), "F1": make_scorer(f1_score)}


iforest_1_eval = RandomizedSearchCV(iforest_1, grid, cv = cv, verbose = 4, random_state=0,\
     scoring = sc, n_iter = 10, refit="PR_AUC_score", n_jobs=2)
iforest_1_tuning = iforest_1_eval.fit(np.array(X_train1), X_labels)
iforest_1 = iforest_1_tuning.best_estimator_
score_iforest1 = iforest_1_tuning.cv_results_

iforest_2_eval = RandomizedSearchCV(iforest_2, grid, cv = cv, verbose = 4, random_state=0,\
     scoring = sc, n_iter = 10, refit="PR_AUC_score", n_jobs=2)
iforest_2_tuning = iforest_2_eval.fit(np.array(X_train2), X_labels)
iforest_2 = iforest_2_tuning.best_estimator_
score_iforest2 = iforest_2_tuning.cv_results_

iforest_3_eval = RandomizedSearchCV(iforest_3, grid, cv = cv, verbose = 4, random_state=0,\
     scoring = sc, n_iter = 10, refit="PR_AUC_score", n_jobs=2)
iforest_3_tuning = iforest_3_eval.fit(np.array(X_train3), X_labels)
iforest_3 = iforest_3_tuning.best_estimator_
score_iforest3 = iforest_3_tuning.cv_results_

# save the model to disk
joblib.dump(iforest_1, 'ML_Models\ML_iforest_1.sav')
joblib.dump(iforest_2, 'ML_Models\ML_iforest_2.sav')
joblib.dump(iforest_3, 'ML_Models\ML_iforest_3.sav')