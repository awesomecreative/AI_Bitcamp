import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# path = './_save/'
# # path = '../_save/'
# # path = 'c:/study/_save/'

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

x_train = x_train.reshape(404, 13, 1)
x_test = x_test.reshape(102, 13, 1)
print(x_train.shape, x_test.shape)

# #2. 모델구성(순차형)
model = Sequential()
model.add(LSTM(64, input_shape=(13,1), activation='relu'))
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

# import datetime
# date = datetime.datetime.now()
# print(date) # 2023-01-12 15:02:53.276504
# print(type(date)) # <class 'datetime.datetime'>
# date = date.strftime("%m%d_%H%M") # 0112_1502
# print(date)
# print(type(date)) # <class 'str'>

# filepath = './_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
#                      filepath = filepath + 'k31_01_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=5000, batch_size=32, validation_split=0.2, callbacks=[es], verbose=1)
# callbacks=[es,mcp]

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
Epoch 118/5000
11/11 [==============================] - ETA: 0s - loss: 29.4053 - mae: 4.0651Restoring model weights from the end of the best epoch: 98.
11/11 [==============================] - 0s 40ms/step - loss: 29.4053 - mae: 4.0651 - val_loss: 37.7207 - val_mae: 3.9737
Epoch 00118: early stopping
=================1. 기본출력 ==================
4/4 [==============================] - 0s 8ms/step - loss: 28.3821 - mae: 3.9769
mse :  28.382095336914062
mae :  3.9768991470336914
RMSE :  5.327485011476917
R2 :  0.6569548874811315
"""