#Main LSTM script
#Libraries
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

np.random.seed(1)
tf.random.set_seed(1)

#Create Data
from DataCreation import TSwithAnomaly
Value = TSwithAnomaly(10, 1, 40, 35, 40, "cyclic")
Sample = np.arange(0, np.size(Value))
TS = pd.DataFrame({"Sample":Sample, "Value":Value})

#Visualize Data
fig = go.Figure()
fig.add_trace(go.Scatter(x=TS['Sample'], y=TS['Value'], name='Synthetic TS'))
fig.update_layout(showlegend=True, title='Synthetic TS')
fig.show()

#Split Data
train, test = TS[0:30], TS[30:40]

#Scale Data
scaler = StandardScaler()
scaler = scaler.fit(train[["Value"]])
train["Value"] = scaler.transform(train[["Value"]])
test["Value"] = scaler.transform(test[["Value"]])

TIME_STEPS=4

def create_sequences(X, y, time_steps = TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])
    
    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(train[['Value']], train['Value'])
X_test, y_test = create_sequences(test[['Value']], test['Value'])

print(f'Training shape: {X_train.shape}')
print(f'Testing shape: {X_test.shape}')

# define model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(rate = 0.2))
model.add(RepeatVector(X_train.shape[1]))
model.add(LSTM(128, return_sequences = True))
model.add(Dropout(rate = 0.2))
model.add(TimeDistributed(Dense(X_train.shape[2])))
model.compile(optimizer = 'adam', loss = 'mae')
model.summary()

# fit model
history = model.fit(X_train, y_train, epochs = 100, batch_size = 1, validation_split = 0.1,\
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')],\
    shuffle = False)

model.evaluate(X_test, y_test)

#anomaly detection
X_train_pred = model.predict(X_train, verbose = 0)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis = 1)
threshold = np.max(train_mae_loss)

X_test_pred = model.predict(X_test, verbose=0)
test_mae_loss = np.mean(np.abs(X_test_pred-X_test), axis=1)


test_score_df = pd.DataFrame(test[TIME_STEPS:])
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = threshold
test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
test_score_df['Value'] = test[TIME_STEPS:]['Value']

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df['Sample'], y=test_score_df['loss'], name='Test loss'))
fig.add_trace(go.Scatter(x=test_score_df['Sample'], y=test_score_df['threshold'], name='Threshold'))
fig.update_layout(showlegend=True, title='Test loss vs. Threshold')
fig.show()

anomalies = test_score_df.loc[test_score_df['anomaly'] == True]

test_score_df["Value"] = scaler.inverse_transform(test_score_df[['Value']])
anomalies["Value"] = scaler.inverse_transform(anomalies[['Value']])

fig = go.Figure()
fig.add_trace(go.Scatter(x=test_score_df['Sample'], y=test_score_df['Value'], name='Value'))
fig.add_trace(go.Scatter(x=anomalies['Sample'], y=anomalies['Value'], mode='markers', name='Anomaly'))
fig.update_layout(showlegend=True, title='Detected anomalies')
fig.show()