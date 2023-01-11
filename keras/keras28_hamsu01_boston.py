# <순차형 모델을 함수형 모델로 바꾸기>

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
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

# #2. 모델구성(순차형)
# model = Sequential()
# model.add(Dense(50, input_dim=13, activation = 'relu'))
# model.add(Dense(40, activation = 'sigmoid'))
# model.add(Dense(30, activation = 'relu'))
# model.add(Dense(20, activation = 'linear'))
# model.add(Dense(1, activation = 'linear'))
# model.summary()

# # Model: "sequential"
# # _________________________________________________________________
# #  Layer (type)                Output Shape              Param #
# # =================================================================
# #  dense (Dense)               (None, 50)                700

# #  dense_1 (Dense)             (None, 40)                2040

# #  dense_2 (Dense)             (None, 30)                1230

# #  dense_3 (Dense)             (None, 20)                620

# #  dense_4 (Dense)             (None, 1)                 21

# # =================================================================
# # Total params: 4,611
# # Trainable params: 4,611
# # Non-trainable params: 0
# # _________________________________________________________________

#2. 모델구성(함수형)
input1 = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(1, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()

# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 13)]              0

#  dense (Dense)               (None, 50)                700

#  dense_1 (Dense)             (None, 40)                2040

#  dense_2 (Dense)             (None, 30)                1230

#  dense_3 (Dense)             (None, 20)                620

#  dense_4 (Dense)             (None, 1)                 21

# =================================================================
# Total params: 4,611
# Trainable params: 4,611
# Non-trainable params: 0
# _________________________________________________________________

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'] )

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[earlyStopping])

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
# <순차형 모델을 함수형 모델로 바꾸기>
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
순차형 모델은 모델의 종류부터 쓰지만 함수형 모델은 레이어를 다 쓰고 마지막에 모델을 명시한다.
input1 = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(1, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1)

함수형의 장점은 layer을 건너 뛰어서 입력할 수 있음.
예를 들어, dense3에 입력되는 layer를 dense1으로 설정가능하다.
"""