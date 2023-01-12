# https://www.kaggle.com/competition/bike-sharing-demand

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
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

# #2. model
# model = Sequential()
# model.add(Dense(10, input_dim=8, activation='sigmoid'))
# model.add(Dense(20, activation='sigmoid'))
# model.add(Dense(40, activation='sigmoid'))
# model.add(Dense(80, activation='sigmoid'))
# model.add(Dropout(0.3))
# model.add(Dense(40, activation='sigmoid'))
# model.add(Dense(20, activation='sigmoid'))
# model.add(Dense(1, activation='linear'))

#2. model
input1 = Input(shape=(8,))
dense1 = Dense(10, activation='sigmoid')(input1)
dense2 = Dense(20, activation='sigmoid')(dense1)
dense3 = Dense(40, activation='sigmoid')(dense2)
dense4 = Dense(80, activation='sigmoid')(dense3)
drop4 = Dropout(0.3)(dense4)
dense5 = Dense(40, activation='sigmoid')(drop4)
dense6 = Dense(20, activation='sigmoid')(dense5)
output1 = Dense(1, activation='linear')(dense6)
model = Model(inputs=input1, outputs=output1)

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
                     filepath = filepath + 'k31_05_' + date + '_' + filename)

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

submission_csv.to_csv(filepath2 + 'submission_save_05_' + date + '.csv')

print('time: ', end-start)
