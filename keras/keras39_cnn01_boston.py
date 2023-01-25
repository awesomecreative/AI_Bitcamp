import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = './_save/'
# path = '../_save/'
# path = 'c:/study/_save/'

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)

print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(404, 13, 1, 1)
x_test = x_test.reshape(102, 13, 1, 1)
print(x_train.shape, x_test.shape)

# #2. 모델구성(순차형)
model = Sequential()
model.add(Conv2D(64, (2,1), input_shape=(13,1,1), activation='relu', padding='same'))
model.add(Conv2D(32, (2,1), activation='relu', padding='same'))
model.add(Flatten())
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='linear'))
model.add(Dense(1, activation='linear'))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'] )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, 
                   restore_best_weights=True,
                   verbose=1)

import datetime
date = datetime.datetime.now()
print(date) # 2023-01-12 15:02:53.276504
print(type(date)) # <class 'datetime.datetime'>
date = date.strftime("%m%d_%H%M") # 0112_1502
print(date)
print(type(date)) # <class 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                     filepath = filepath + 'k31_01_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=5000, batch_size=32, validation_split=0.2, callbacks=[es, mcp], verbose=1)

# model.save(path + 'keras30_ModelCheckPoint3_save_model.h5')


#4. 평가, 예측
print("=================1. 기본출력 ==================")
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

"""

print("=================2. load_model 출력 ==================")
model2 = load_model(path + 'keras30_ModelCheckPoint3_save_model.h5')
mse, mae = model2.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model2.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

print("=================3. ModelCheckPoint 출력 ==================")
model3 = load_model(path + 'MCP/keras30_ModelCheckPoint3.hdf5')
mse, mae = model3.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model3.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

"""

"""
Epoch 81/5000
 1/11 [=>............................] - ETA: 0s - loss: 12.8274 - mae: 2.9912Restoring model weights from the end of the best epoch: 61.

Epoch 00081: val_loss did not improve from 18.71975
11/11 [==============================] - 0s 7ms/step - loss: 18.0395 - mae: 3.1940 - val_loss: 31.3521 - val_mae: 4.1602
Epoch 00081: early stopping
=================1. 기본출력 ==================
4/4 [==============================] - 0s 15ms/step - loss: 24.8278 - mae: 2.8440
mse :  24.82783317565918
mae :  2.8439691066741943
RMSE :  4.982753301184029
R2 :  0.6999141384904191
"""

"""
<학습 내용>
■ 일반 데이터를 CNN으로 바꾸기
(404, 13) => (13,1,1)로 reshape (세로 13개, 가로 1개, 흑백)
layer에 Conv2D, Flatten 추가
kernel_size=(2,2)불가 (2,1)가능 (13,1,1) 이기 때문
# (N, 8) = (N, 2, 2, 2) = (N, 4, 2, 1) = (N, 8, 1, 1) 다 가능함.
# Scaler 먼저 하고 reshape 하기
"""