# https://www.kaggle.com/competition/bike-sharing-demand

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. data
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
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=300, verbose=1, restore_best_weights=True)
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

#5. draw, submit
import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('bike loss')
plt.legend(loc='center right')
plt.show()

y_submit = model.predict(test_csv)
submission_csv['count'] = y_submit

submission_csv.to_csv(path + 'submission_ear_010904.csv')
print ('RMSE : ', rmse)
print('R2 : ', r2)
print('time: ', end-start)

"""
submission_ear_010901.csv
Epoch 88/1000
173/218 [======================>.......] - ETA: 0s - loss: 32818.1719 - mae: 141.9352 - mse: 32818.1719Restoring model weights from the end of the best epoch: 83.
218/218 [==============================] - 0s 1ms/step - loss: 32755.1074 - mae: 142.0152 - mse: 32755.1074 - val_loss: 31363.9473 - val_mae: 140.6048 - val_mse: 31363.9473
Epoch 00088: early stopping
69/69 [==============================] - 0s 580us/step - loss: 34166.2461 - mae: 144.9258 - mse: 34166.2461
loss :  [34166.24609375, 144.92578125, 34166.24609375]
RMSE:  184.84115236963154
r2:  -0.001938932970901197

submission_ear_010902.csv : patience=50
Epoch 385/1000
170/218 [======================>.......] - ETA: 0s - loss: 22536.2852 - mae: 111.0825 - mse: 22536.2852Restoring model weights from the end of the best epoch: 335.
218/218 [==============================] - 0s 1ms/step - loss: 22489.6133 - mae: 111.0797 - mse: 22489.6133 - val_loss: 22378.7188 - val_mae: 110.7743 - val_mse: 22378.7188
Epoch 00385: early stopping
69/69 [==============================] - 0s 586us/step - loss: 23836.5879 - mae: 112.8646 - mse: 23836.5879
loss :  [23836.587890625, 112.8646240234375, 23836.587890625]
RMSE:  154.39102601004828
r2:  0.30098253874395253

submission_ear_010903.csv : patience=100
Epoch 593/5000
203/218 [==========================>...] - ETA: 0s - loss: 22864.6719 - mae: 111.5972 - mse: 22864.6719Restoring model weights from the end of the best epoch: 493.
218/218 [==============================] - 0s 954us/step - loss: 22587.1797 - mae: 110.9928 - mse: 22587.1797 - val_loss: 22059.5664 - val_mae: 111.9826 - val_mse: 22059.5664
Epoch 00593: early stopping
69/69 [==============================] - 0s 500us/step - loss: 23881.4043 - mae: 112.6106 - mse: 23881.4043
loss :  [23881.404296875, 112.61061096191406, 23881.404296875]
RMSE:  154.53608456040158
r2:  0.2996683937857103

submission_ear_010904.csv : patience=300
Epoch 1474/10000
167/218 [=====================>........] - ETA: 0s - loss: 21830.6758 - mae: 109.0012 - mse: 21830.6758Restoring model weights from the end of the best epoch: 1174.
218/218 [==============================] - 0s 1ms/step - loss: 21773.3359 - mae: 108.7961 - mse: 21773.3359 - val_loss: 22031.0996 - val_mae: 110.2552 - val_mse: 22031.0996
Epoch 01474: early stopping
69/69 [==============================] - 0s 561us/step - loss: 23910.7637 - mae: 112.3350 - mse: 23910.7637
loss :  [23910.763671875, 112.33499145507812, 23910.763671875]
RMSE:  154.63105800401195
r2:  0.29880732184950964
"""
