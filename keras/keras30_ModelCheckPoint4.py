import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
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

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델구성(순차형)
# model = Sequential()
# model.add(Dense(50, input_dim=13, activation = 'relu'))
# model.add(Dense(40, activation = 'sigmoid'))
# model.add(Dense(30, activation = 'relu'))
# model.add(Dense(20, activation = 'linear'))
# model.add(Dense(1, activation = 'linear'))
# model.summary()

#2. 모델구성(함수형)
input1 = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(1, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1)
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

# epoch의 점수 4자리, loss의 소수점 4자리까지 있는 모델 파일명
# {} 안에 있는 건 값 가져오라는 뜻임. 일반적인 문자와 다름.
# s String 문자형
# d Decimal 숫자형
# f Float 소수형
# 0nd : n자리 숫자 ex) 04d 하면 0001 0002 0003
# d : 공백 없이 그냥 숫자 ex) d 하면 1 2 3 4 5 6 7 ...
# nd : 공백 있게 숫자 ex) 4d 하면 공백공백공백1 공백공백10, 공백100, 1000 이런 식임
# .nf : n자리 소수
# ModelCheckpoint는 fit할 때 하므로 epoch랑 val_loss 다 저장되어 있음.

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                     # filepath= path + 'MCP/keras30_ModelCheckPoint3.hdf5')
                     filepath = filepath + 'k30_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=5000, batch_size=32, validation_split=0.2, callbacks=[es, mcp], verbose=1)

# model.save(path + 'keras30_ModelCheckPoint3_save_model.h5')

"""
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