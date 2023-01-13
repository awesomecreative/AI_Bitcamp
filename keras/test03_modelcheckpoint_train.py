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
                   restore_best_weights=False,
                   verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                      filepath= path + 'MCP/keras30_ModelCheckPoint4.hdf5')

model.fit(x_train, y_train, epochs=5000, batch_size=32, validation_split=0.2, callbacks=[es, mcp], verbose=1)

model.save(path + 'keras30_ModelCheckPoint4_save_model.h5')

#4. 평가, 예측
print("=================1. 기본출력 ==================")
mse, mae = model.evaluate(x_train, y_train)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model.predict(x_train)

def RMSE(y_train, y_predict):
    return np.sqrt(mean_squared_error(y_train, y_predict))
print("RMSE : ", RMSE(y_train, y_predict))

r2 = r2_score(y_train, y_predict)
print("R2 : ", r2)

print("=================2. load_model 출력 ==================")
model2 = load_model(path + 'keras30_ModelCheckPoint4_save_model.h5')
mse, mae = model2.evaluate(x_train, y_train)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model2.predict(x_train)

def RMSE(y_train, y_predict):
    return np.sqrt(mean_squared_error(y_train, y_predict))
print("RMSE : ", RMSE(y_train, y_predict))

r2 = r2_score(y_train, y_predict)
print("R2 : ", r2)


print("=================3. ModelCheckPoint 출력 ==================")
model3 = load_model(path + 'MCP/keras30_ModelCheckPoint4.hdf5')
mse, mae = model3.evaluate(x_train, y_train)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model3.predict(x_train)

def RMSE(y_train, y_predict):
    return np.sqrt(mean_squared_error(y_train, y_predict))
print("RMSE : ", RMSE(y_train, y_predict))

r2 = r2_score(y_train, y_predict)
print("R2 : ", r2)

"""
엥...?!

Epoch 138/5000
 1/11 [=>............................] - ETA: 0s - loss: 3.7734 - mae: 1.4594
Epoch 00138: val_loss did not improve from 14.56477
11/11 [==============================] - 0s 2ms/step - loss: 4.5887 - mae: 1.5727 - val_loss: 15.7261 - val_mae: 2.2651
Epoch 00138: early stopping
=================1. 기본출력 ==================
13/13 [==============================] - 0s 578us/step - loss: 6.6239 - mae: 1.6917
mse :  6.6239213943481445
mae :  1.69170343875885
RMSE :  2.5736981350540638
R2 :  0.9218146417266712
=================2. load_model 출력 ==================
13/13 [==============================] - 0s 605us/step - loss: 6.6239 - mae: 1.6917
mse :  6.6239213943481445
mae :  1.69170343875885
RMSE :  2.5736981350540638
R2 :  0.9218146417266712
=================3. ModelCheckPoint 출력 ==================
13/13 [==============================] - 0s 585us/step - loss: 6.8022 - mae: 1.7587
mse :  6.802216053009033
mae :  1.758730173110962
RMSE :  2.608105814838017
R2 :  0.9197101531126703

x_train을 넣어서 평가하고 예측했는데도 ModelCheckPoint가 r2가 낮게 나왔다...!

7번째 계속 낮게 나온다...
"""

