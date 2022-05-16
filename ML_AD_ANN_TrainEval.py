#Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.ensemble import IsolationForest
from sklearn import svm
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
split= int(3/5 * X_train1.shape[0])

X_train1 = X_train1.iloc[0:split, :]
X_train2 = X_train2.iloc[0:split, :]

#Structuring labels
TS1_Class = pd.read_csv("TimeSeries1_Classification.csv", index_col = 0)
TS1_Class = TS1_Class.iloc[0:960, 20:60]
X_labels = np.array(TS1_Class).reshape(TS1_Class.size)

#ANN trial
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import gradient_descent_v2


hidden_units=100
learning_rate=0.01
hidden_layer_act='tanh'
output_layer_act='sigmoid'
no_epochs=100

ANN_2 = Sequential()
ANN_2.add(Dense(hidden_units, input_dim=7, activation=hidden_layer_act))
ANN_2.add(Dense(hidden_units, activation=hidden_layer_act))
ANN_2.add(Dense(1, activation=output_layer_act))
sgd = gradient_descent_v2.SGD(lr = learning_rate)
ANN_2.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['acc'])

ANN_2.fit(X_train2, X_labels, epochs=no_epochs, batch_size=len(X_train2),  verbose=2)

ANN_2.save('ML_ANN_2.h5')