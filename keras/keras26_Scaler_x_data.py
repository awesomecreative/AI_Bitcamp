# <scaler>

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

# ############민멕스 스케일러#############
# scaler = MinMaxScaler()
# scaler.fit(x) # 가중치 설정할 뿐 x에 저장하진 않음.
# x = scaler.transform(x) # 나온 가중치로 x를 바꾸는 게 transform임.

# print(x, x.shape, type(x))
# print(np.min(x), np.max(x)) # 0.0 1.0
# #######################################

# ###########스탠다드 스케일러#############
# scaler = StandardScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# ########################################

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=13, activation = 'linear'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'] )
model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.2)

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

"""
<MinMax Scaler>
Epoch 500/500
11/11 [==============================] - 0s 3ms/step - loss: 2.5213 - mae: 1.1943 - val_loss: 16.5976 - val_mae: 2.4025
4/4 [==============================] - 0s 1ms/step - loss: 20.1456 - mae: 2.6049
mse :  20.145593643188477
mae :  2.604896068572998
RMSE :  4.488384319730258
R2 :  0.7565068006565058

<Standard Scaler>
Epoch 500/500
11/11 [==============================] - 0s 3ms/step - loss: 0.7148 - mae: 0.6534 - val_loss: 17.7643 - val_mae: 2.5725
4/4 [==============================] - 0s 997us/step - loss: 18.0454 - mae: 2.5346
mse :  18.045427322387695
mae :  2.534555435180664
RMSE :  4.247990830824106
R2 :  0.781890840370465
"""

"""
<MinMaxScaler>
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x) # 가중치 설정할 뿐 x에 저장하진 않음.
x = scaler.transform(x) # 나온 가중치로 x를 바꾸는 게 transform임.
print(x, x.shape, type(x))
print(np.min(x), np.max(x)) # 0.0 1.0

<StandardScaler>
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
"""