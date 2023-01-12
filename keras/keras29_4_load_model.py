# <모델, 가중치 불러오는 방법>

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

print(x, x.shape, type(x))
print(np.min(x), np.max(x)) # 0.0 711.0

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


path = './_save/'
# path = '../_save/'
# path = 'c:/study/_save/'

# model.save(path + 'keras29_1_save_model.h5')
# model.save('./_save/keras29_1_save_model.h5')

model = load_model(path + 'keras29_3_save_model.h5')

#3. 컴파일, 훈련
# Epoch 68/1000
#  1/11 [=>............................] - ETA: 0s - loss: 10.1403 - mae: 2.4589Restoring model weights from the end of the best epoch: 58.
# 11/11 [==============================] - 0s 2ms/step - loss: 7.0503 - mae: 1.9309 - val_loss: 16.1955 - val_mae: 2.3539
# Epoch 00068: early stopping
# 4/4 [==============================] - 0s 665us/step - loss: 20.2902 - mae: 2.7578
# mse :  20.29020881652832
# mae :  2.7577576637268066
# RMSE :  4.504465238045875
# R2 :  0.7547589072780435

# 4/4 [==============================] - 0s 0s/step - loss: 20.2902 - mae: 2.7578
# mse :  20.29020881652832
# mae :  2.7577576637268066
# RMSE :  4.504465238045875
# R2 :  0.7547589072780435

#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)