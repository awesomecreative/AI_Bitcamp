import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
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

# #2. 모델구성(함수형)
input1 = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(1, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()

# model.save_weights(path + 'keras29_5_save_weights1.h5')
# model.load_weights(path + 'keras29_5_save_weights1.h5')
model.load_weights(path + 'keras29_5_save_weights1.h5')

"""
save_weights는 모델 저장은 안되고 순수하게 가중치만 저장된다.
컴파일, 훈련 전에 save_weights 하면 가중치가 아예 없음.
컴파일, 훈련 이후에 save_weights 하면 컴파일, 훈련된 가중치만 저장된다.

컴파일 하지 않고 load_weights 하면
You must compile your model before training/testing. Use `model.compile(optimizer, loss)` 에러 뜸.

load_weights 쓰려면 모델이랑 컴파일까지 알아야 함.
"""

# # #3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'] )

# from tensorflow.keras.callbacks import EarlyStopping
# earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
# model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[earlyStopping], verbose=1)

# # model.save_weights(path + 'keras29_5_save_weights2.h5')

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
Epoch 86/1000
 1/11 [=>............................] - ETA: 0s - loss: 4.8498 - mae: 1.8985Restoring model weights from the end of the best epoch: 76.
11/11 [==============================] - 0s 3ms/step - loss: 5.8361 - mae: 1.7698 - val_loss: 17.9023 - val_mae: 2.4458
Epoch 00086: early stopping
4/4 [==============================] - 0s 949us/step - loss: 21.0521 - mae: 2.6850
mse :  21.052125930786133
mae :  2.6850013732910156
RMSE :  4.588259421618389
R2 :  0.7455498605376323
"""