# https://www.kaggle.com/competition/bike-sharing-demand

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. data
# path = './_data/bike/'
path = 'c:/study/_data/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.shape)

x = train_csv.drop(['casual','registered','count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

print(x_train.shape, x_test.shape, test_csv.shape) # (8708, 8) (2178, 8) (6493, 8)

x_train = x_train.reshape(8708, 2, 2, 2)
x_test = x_test.reshape(2178, 2, 2, 2)
test_csv = test_csv.reshape(6493, 2, 2, 2)

#2. model
model = Sequential()
model.add(Conv2D(512, (2,2), input_shape=(2,2,2), padding='same', activation='relu'))
model.add(Conv2D(128, (2,2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))


#3. compile, fit
import time
model.compile(loss='mse', optimizer='adam', metrics=['mae','mse'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True)


import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                     filepath = filepath + 'k39_bi_' + date + '_' + filename)

start = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=1,
                 callbacks=[es, mcp])
end = time.time()

#4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('RMSE: ', rmse)
print('r2: ', r2)

# #5. draw, submit
# import matplotlib.pyplot as plt

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
# plt.grid()
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title('bike loss')
# plt.legend(loc='center right')
# plt.show()

y_submit = model.predict(test_csv)

submission_csv['count'] = y_submit

filepath2 = 'c:/study/_data/bike/'

submission_csv.to_csv(filepath2 + 'submission_conv2d_' + date + '.csv')

print('time: ', end-start)

"""
Epoch 88/1000
212/218 [============================>.] - ETA: 0s - loss: 14491.7041 - mae: 86.3572 - mse: 14491.7041Restoring model weights from the end of the best epoch: 38.

Epoch 00088: val_loss did not improve from 20290.19922
218/218 [==============================] - 1s 5ms/step - loss: 14496.9014 - mae: 86.3268 - mse: 14496.9014 - val_loss: 21842.2773 - val_mae: 107.1637 - val_mse: 21842.2773
Epoch 00088: early stopping
69/69 [==============================] - 0s 3ms/step - loss: 22921.4180 - mae: 108.9223 - mse: 22921.4180
loss :  [22921.41796875, 108.9223403930664, 22921.41796875]
RMSE:  151.39822903070007
r2:  0.32782012143191774
time:  111.78694009780884
"""