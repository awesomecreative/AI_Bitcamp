# https://www.kaggle.com/competition/bike-sharing-demand

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. data
# path = './_data/bike/'
path = 'c:/study/_data/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.shape)

x = train_csv.drop(['casual','registered','count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

#2. model
model = Sequential()
model.add(Dense(10, input_dim=8, activation='sigmoid'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(80, activation='sigmoid'))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(20, activation='sigmoid'))
model.add(Dense(1, activation='linear'))

#3. compile, fit
import time
model.compile(loss='mse', optimizer='adam', metrics=['mae','mse'])
start = time.time()
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=10000, batch_size=32, validation_split=0.2, verbose=1,
                 callbacks=[earlyStopping])
end = time.time()

#4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('RMSE: ', rmse)
print('r2: ', r2)

# #5. draw, submit
# import matplotlib.pyplot as plt

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
# plt.grid()
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title('bike loss')
# plt.legend(loc='center right')
# plt.show()

y_submit = model.predict(test_csv)
submission_csv['count'] = y_submit

submission_csv.to_csv(path + 'submission_scal_011108.csv')
print ('RMSE : ', rmse)
print('R2 : ', r2)
print('time: ', end-start)

"""
<test_csv 전처리 안해줘서 망한 데이터...>
1) scaling 안 했을 때
submission_ear_010902.csv : patience=50
Epoch 385/1000
170/218 [======================>.......] - ETA: 0s - loss: 22536.2852 - mae: 111.0825 - mse: 22536.2852Restoring model weights from the end of the best epoch: 335.
218/218 [==============================] - 0s 1ms/step - loss: 22489.6133 - mae: 111.0797 - mse: 22489.6133 - val_loss: 22378.7188 - val_mae: 110.7743 - val_mse: 22378.7188
Epoch 00385: early stopping
69/69 [==============================] - 0s 586us/step - loss: 23836.5879 - mae: 112.8646 - mse: 23836.5879
loss :  [23836.587890625, 112.8646240234375, 23836.587890625]
RMSE:  154.39102601004828
r2:  0.30098253874395253

2) MinMaxScaler
submission_scal_011101.csv
Epoch 1135/10000
213/218 [============================>.] - ETA: 0s - loss: 22068.8496 - mae: 110.4027 - mse: 22068.8496Restoring model weights from the end of the best epoch: 1085.
218/218 [==============================] - 0s 1ms/step - loss: 21970.1836 - mae: 110.1729 - mse: 21970.1836 - val_loss: 21613.7461 - val_mae: 109.2898 - val_mse: 21613.7461
Epoch 01135: early stopping
69/69 [==============================] - 0s 554us/step - loss: 23690.4570 - mae: 112.8447 - mse: 23690.4570
loss :  [23690.45703125, 112.84465026855469, 23690.45703125]
RMSE:  153.91705547522224
r2:  0.3052678282799195
time:  264.72310304641724

submission_scal_011102.csv
Epoch 1064/10000
169/218 [======================>.......] - ETA: 0s - loss: 21427.2227 - mae: 107.8769 - mse: 21427.2227Restoring model weights from the end of the best epoch: 1014.
218/218 [==============================] - 0s 1ms/step - loss: 21381.9258 - mae: 107.5259 - mse: 21381.9258 - val_loss: 21262.1074 - val_mae: 109.1572 - val_mse: 21262.1074
Epoch 01064: early stopping
69/69 [==============================] - 0s 601us/step - loss: 22958.7109 - mae: 109.7098 - mse: 22958.7109
loss :  [22958.7109375, 109.7098388671875, 22958.7109375]
RMSE:  151.52132076050466
r2:  0.3267266684952108

3) StandardScaler
submission_scal_011103.csv
Epoch 421/10000
170/218 [======================>.......] - ETA: 0s - loss: 21536.8945 - mae: 107.8293 - mse: 21536.8945Restoring model weights from the end of the best epoch: 371.
218/218 [==============================] - 0s 1ms/step - loss: 21643.5664 - mae: 108.4192 - mse: 21643.5664 - val_loss: 21485.6465 - val_mae: 109.4078 - val_mse: 21485.6465
Epoch 00421: early stopping
69/69 [==============================] - 0s 566us/step - loss: 23748.2051 - mae: 113.0164 - mse: 23748.2051
loss :  [23748.205078125, 113.01640319824219, 23748.205078125]
RMSE:  154.10454982094882
r2:  0.30357421881192637
time:  105.006103515625

submission_scal_011104.csv
Epoch 393/10000
174/218 [======================>.......] - ETA: 0s - loss: 21436.8613 - mae: 108.3008 - mse: 21436.8613Restoring model weights from the end of the best epoch: 343.
218/218 [==============================] - 0s 1ms/step - loss: 21753.8379 - mae: 108.8385 - mse: 21753.8379 - val_loss: 21632.0957 - val_mae: 110.8227 - val_mse: 21632.0957
Epoch 00393: early stopping
69/69 [==============================] - 0s 660us/step - loss: 23778.0781 - mae: 112.6917 - mse: 23778.0781
loss :  [23778.078125, 112.6916732788086, 23778.078125]
RMSE:  154.20142926573604
r2:  0.3026983128019973
time:  99.78455686569214

: Kaggle_bike 에서는 다 비슷하게 나왔다.
"""


"""
<test_csv 전처리 해줌!>
1) scaling 안 했을 때
submission_ear_010902.csv : patience=50
Epoch 385/1000
170/218 [======================>.......] - ETA: 0s - loss: 22536.2852 - mae: 111.0825 - mse: 22536.2852Restoring model weights from the end of the best epoch: 335.
218/218 [==============================] - 0s 1ms/step - loss: 22489.6133 - mae: 111.0797 - mse: 22489.6133 - val_loss: 22378.7188 - val_mae: 110.7743 - val_mse: 22378.7188
Epoch 00385: early stopping
69/69 [==============================] - 0s 586us/step - loss: 23836.5879 - mae: 112.8646 - mse: 23836.5879
loss :  [23836.587890625, 112.8646240234375, 23836.587890625]
RMSE:  154.39102601004828
r2:  0.30098253874395253

2) MinMaxScaler
submission_scal_011105.csv
Epoch 891/10000
203/218 [==========================>...] - ETA: 0s - loss: 22074.6504 - mae: 109.5326 - mse: 22074.6504Restoring model weights from the end of the best epoch: 841.
218/218 [==============================] - 0s 1ms/step - loss: 21889.3965 - mae: 109.3562 - mse: 21889.3965 - val_loss: 21483.9883 - val_mae: 108.1731 - val_mse: 21483.9883
Epoch 00891: early stopping
69/69 [==============================] - 0s 631us/step - loss: 23402.0957 - mae: 111.5968 - mse: 23402.0957
loss :  [23402.095703125, 111.59681701660156, 23402.095703125]
RMSE:  152.97743636428808
r2:  0.3137242153371572
time:  213.99106574058533

submission_scal_011106.csv
Epoch 754/10000
212/218 [============================>.] - ETA: 0s - loss: 22135.2930 - mae: 110.7681 - mse: 22135.2930Restoring model weights from the end of the best epoch: 704.
218/218 [==============================] - 1s 4ms/step - loss: 22309.9727 - mae: 111.1832 - mse: 22309.9727 - val_loss: 22043.2832 - val_mae: 113.8536 - val_mse: 22043.2832
Epoch 00754: early stopping
69/69 [==============================] - 0s 2ms/step - loss: 23877.2480 - mae: 113.6268 - mse: 23877.2480
loss :  [23877.248046875, 113.62684631347656, 23877.248046875]
RMSE:  154.52263941386482
r2:  0.29979025078089094
time:  791.0129444599152

3) StandardScaler
submission_scal_011107.csv
Epoch 392/10000
169/218 [======================>.......] - ETA: 0s - loss: 21809.9258 - mae: 108.3537 - mse: 21809.9258Restoring model weights from the end of the best epoch: 342.
218/218 [==============================] - 0s 1ms/step - loss: 21790.7832 - mae: 108.8103 - mse: 21790.7832 - val_loss: 21513.4395 - val_mae: 108.7351 - val_mse: 21513.4395
Epoch 00392: early stopping
69/69 [==============================] - 0s 704us/step - loss: 23849.7578 - mae: 113.3461 - mse: 23849.7578
loss :  [23849.7578125, 113.34613037109375, 23849.7578125]
RMSE:  154.4336636405384
r2:  0.3005963950089864
time:  95.98977541923523

submission_scal_011108.csv
Epoch 599/10000
208/218 [===========================>..] - ETA: 0s - loss: 21461.7188 - mae: 108.2066 - mse: 21461.7188Restoring model weights from the end of the best epoch: 549.
218/218 [==============================] - 0s 1ms/step - loss: 21524.8105 - mae: 108.5547 - mse: 21524.8105 - val_loss: 21465.4785 - val_mae: 110.8327 - val_mse: 21465.4785
Epoch 00599: early stopping
69/69 [==============================] - 0s 675us/step - loss: 23864.4160 - mae: 113.2327 - mse: 23864.4160
loss :  [23864.416015625, 113.23274993896484, 23864.416015625]
RMSE:  154.4811270999243
r2:  0.300166421193117

: Kaggle_bike 에서는 다 비슷하게 나왔다.
"""

