# https://www.kaggle.com/competition/bike-sharing-demand

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.shape)

x = train_csv.drop(['casual','registered','count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=8, activation='sigmoid'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(80, activation='sigmoid'))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam', metrics=['mae','mse'])
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)


def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

y_submit = model.predict(test_csv)
submission_csv['count'] = y_submit

submission_csv.to_csv(path + 'submission_val_010605.csv')
print ('RMSE : ', rmse)
print('R2 : ', r2)
print('time: ', end-start)

"""
submission_val_01060447.csv
Epoch 500/500
218/218 [==============================] - 0s 1ms/step - loss: 17052.0703 - mae: 95.2268 - mse: 17052.0703 - val_loss: 23780.0664 - val_mae: 113.9511 - val_mse: 23780.0664
69/69 [==============================] - 0s 560us/step - loss: 25295.3672 - mae: 115.6349 - mse: 25295.3672
RMSE :  159.04515526368215
R2 :  0.2096481511399877
time:  120.60942244529724

submission_val_010602.csv
Epoch 500/500
218/218 [==============================] - 0s 986us/step - loss: 19822.5645 - mae: 103.6057 - mse: 19822.5645 - val_loss: 21543.8730 - val_mae: 111.4947 - val_mse: 21543.8730
69/69 [==============================] - 0s 526us/step - loss: 22154.7715 - mae: 109.8689 - mse: 22154.7715
RMSE :  148.84478336443684
R2 :  0.30777573100784894
time:  110.32352232933044

submission_val_010603.csv
Epoch 500/500
218/218 [==============================] - 0s 1ms/step - loss: 21574.3047 - mae: 108.6758 - mse: 21574.3047 - val_loss: 21946.1074 - val_mae: 112.7346 - val_mse: 21946.1074
69/69 [==============================] - 0s 506us/step - loss: 22757.2461 - mae: 111.8861 - mse: 22757.2461
RMSE :  150.8550552836127
R2 :  0.288951341522361
time:  113.83789563179016

submission_val_010604.csv
Epoch 500/500
218/218 [==============================] - 0s 1ms/step - loss: 15925.1348 - mae: 91.2899 - mse: 15925.1348 - val_loss: 23339.9707 - val_mae: 111.2776 - val_mse: 23339.9707
69/69 [==============================] - 0s 553us/step - loss: 24419.3789 - mae: 112.2637 - mse: 24419.3789
RMSE :  156.2670319678829
R2 :  0.23701796589827862

submission_val_010605.csv
Epoch 1000/1000
218/218 [==============================] - 0s 1ms/step - loss: 21671.1504 - mae: 108.7895 - mse: 21671.1504 - val_loss: 21675.4805 - val_mae: 111.8606 - val_mse: 21675.4805
69/69 [==============================] - 0s 567us/step - loss: 23755.1855 - mae: 114.4068 - mse: 23755.1855
RMSE :  154.12717344203023
R2 :  0.3033697234933327
time:  224.55982279777527
"""