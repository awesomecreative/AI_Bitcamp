import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
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
# model.add(Dropout(0.5))
# model.add(Dense(40, activation = 'sigmoid'))
# model.add(Dropout(0.3))
# model.add(Dense(30, activation = 'relu'))
# model.add(Dropout(0.2))
# model.add(Dense(20, activation = 'linear'))
# model.add(Dense(1, activation = 'linear'))
# model.summary()

#2. 모델구성(함수형)
input1 = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(input1)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(40, activation='sigmoid')(drop1)
drop2 = Dropout(0.3)(dense2)
dense3 = Dense(30, activation='relu')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(20, activation='linear')(drop3)
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

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                     filepath = filepath + 'k31_01_' + date + '_' + filename)

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

"""
<드롭아웃 Drop-out>
과적합을 막기 위해, 편향되지 않은 출력값을 내기 위해 0~1 사이의 확률로 중간 중간의 노드를 제거해주는 과정을 의미한다.
https://heytech.tistory.com/127

<Dropout 하는 방법>
<순차형 모델>
from tensorflow.keras.layers import Dropout
model.add(Dropout(0~1사이값:확률))

<함수형 모델>
from tensorflow.keras.layers import Dropout
drop1 = Dropout(0~1사이값:확률))(dense1)
dense2 = Dense(40, activation='sigmoid')(drop1)

<주의>
Dropout은 훈련할 때만 적용된다.
evaluate 평가할 때는 모든 데이터를 다 활용한다.
predict 예측할 때는 훈련해서 만들어진 함수, 가중치에 집어넣어서 값이 나오기 때문에 Dropout이 적용안된다.
"""
