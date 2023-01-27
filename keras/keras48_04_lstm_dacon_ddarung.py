import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. data
# path = './_data/ddarung/'
path = 'c:/study/_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv.isnull().sum())
train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train.shape, x_test.shape, test_csv.shape) # (1062, 9) (266, 9) (715, 9)

x_train = x_train.reshape(1062, 9, 1)
x_test = x_test.reshape(266, 9, 1)
test_csv = test_csv.reshape(715, 9, 1)

#2. model
model = Sequential()
model.add(LSTM(512, input_shape=(9,1), activation='relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))


#3. compile, fit : earlystop, time
model.compile(loss='mse', optimizer='adam', metrics=['mae','mse'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=300, verbose=1, restore_best_weights=True)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                     filepath = filepath + 'LSTM_dd_' + date + '_' + filename)

import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=10000, batch_size=16, validation_split=0.2, verbose=1,
                 callbacks=[es, mcp])
end = time.time()

#4. evaluate, predict : RMSE, r2
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('RMSE: ', rmse)
print('r2: ', r2)

#5. draw, submit

# import matplotlib.pyplot as plt

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
# plt.grid()
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title('ddarung loss')
# plt.legend(loc='center right')
# plt.show()

y_submit = model.predict(test_csv)

submission['count'] = y_submit

filepath2 = 'c:/study/_data/ddarung/'
submission.to_csv(filepath2 + 'submission_LSTM_' + date + '.csv')

print('time: ', end-start)

"""
54/54 [==============================] - 3s 56ms/step - loss: 6.1828 - mae: 1.7491 - mse: 6.1828 - val_loss: 1564.1407 - val_mae: 26.4872 - val_mse: 1564.1407
Epoch 1027/10000
54/54 [==============================] - ETA: 0s - loss: 5.4907 - mae: 1.6385 - mse: 5.4907Restoring model weights from the end of the best epoch: 727.

Epoch 01027: val_loss did not improve from 1396.58289
54/54 [==============================] - 3s 55ms/step - loss: 5.4907 - mae: 1.6385 - mse: 5.4907 - val_loss: 1591.6598 - val_mae: 26.7305 - val_mse: 1591.6598
Epoch 01027: early stopping
9/9 [==============================] - 0s 15ms/step - loss: 1445.0673 - mae: 24.4779 - mse: 1445.0673
loss :  [1445.0672607421875, 24.477861404418945, 1445.0672607421875]
RMSE:  38.014042231870924
r2:  0.7513272765321017
time:  1787.861648797989
"""