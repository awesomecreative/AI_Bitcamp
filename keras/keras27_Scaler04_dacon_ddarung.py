import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. data
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv.isnull().sum())
train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis=1)
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
model.add(Dense(20, input_dim=9, activation = 'linear'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(80, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

#3. compile, fit : earlystop, time
model.compile(loss='mse', optimizer='adam', metrics=['mae','mse'])

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=300, verbose=1, restore_best_weights=True)

import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=10000, batch_size=16, validation_split=0.2, verbose=1,
                 callbacks=[earlyStopping])
end = time.time()

#4. evaluate, predict : RMSE, r2
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

# import matplotlib.pyplot as plt

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
# plt.grid()
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title('ddarung loss')
# plt.legend(loc='center right')
# plt.show()

y_submit = model.predict(test_csv)

submission['count'] = y_submit
submission.to_csv(path + 'submission_scal_011108.csv')
print ('RMSE : ', rmse)
print ('r2: ', r2)
print('time: ', end-start)

"""
<test_csv를 scaling 안 해서 submission은 엉망으로 된 데이터...>
1) scaling 안 했을 때
submission_ear_010904.csv : patience=300
Epoch 1583/10000
 1/54 [..............................] - ETA: 0s - loss: 731.1029 - mae: 20.2387 - mse: 731.1029Restoring model weights from the end of the best epoch: 1283.
54/54 [==============================] - 0s 1ms/step - loss: 680.0212 - mae: 16.9566 - mse: 680.0212 - val_loss: 2318.4683 - val_mae: 35.4183 - val_mse: 2318.4683
Epoch 01583: early stopping
9/9 [==============================] - 0s 620us/step - loss: 2078.1926 - mae: 32.7382 - mse: 2078.1926
loss :  [2078.192626953125, 32.73818588256836, 2078.192626953125]
RMSE:  45.58719855758717
r2:  0.6423766605731327

2) MinMaxScaler
submission_scal_011101.csv
Epoch 403/10000
 1/54 [..............................] - ETA: 0s - loss: 233.4823 - mae: 11.5906 - mse: 233.4823Restoring model weights from the end of the best epoch: 103.
54/54 [==============================] - 0s 1ms/step - loss: 284.1885 - mae: 11.2022 - mse: 284.1885 - val_loss: 2254.1646 - val_mae: 32.6720 - val_mse: 2254.1646
Epoch 00403: early stopping
9/9 [==============================] - 0s 764us/step - loss: 1488.2760 - mae: 26.8744 - mse: 1488.2760
loss :  [1488.2760009765625, 26.874412536621094, 1488.2760009765625]
RMSE:  38.578177568230444
r2:  0.7438918137269968
time:  29.648645877838135

submission_scal_011102.csv
Epoch 561/10000
 1/54 [..............................] - ETA: 0s - loss: 52.3168 - mae: 5.4649 - mse: 52.3168Restoring model weights from the end of the best epoch: 261.
54/54 [==============================] - 0s 1ms/step - loss: 111.1139 - mae: 6.5762 - mse: 111.1139 - val_loss: 1845.9287 - val_mae: 29.8183 - val_mse: 1845.9287
Epoch 00561: early stopping
9/9 [==============================] - 0s 623us/step - loss: 1506.3585 - mae: 27.2051 - mse: 1506.3585
loss :  [1506.3585205078125, 27.20509910583496, 1506.3585205078125]
RMSE:  38.81183588884994
r2:  0.7407800528363219
time:  42.35983443260193

3) StandardScaler
submission_scal_011103.csv
Epoch 353/10000
 1/54 [..............................] - ETA: 0s - loss: 76.8949 - mae: 5.5781 - mse: 76.8949Restoring model weights from the end of the best epoch: 53.
54/54 [==============================] - 0s 1ms/step - loss: 45.3446 - mae: 4.7770 - mse: 45.3446 - val_loss: 2200.6506 - val_mae: 32.8715 - val_mse: 2200.6506
Epoch 00353: early stopping
9/9 [==============================] - 0s 623us/step - loss: 1897.8345 - mae: 30.7175 - mse: 1897.8345
loss :  [1897.83447265625, 30.717458724975586, 1897.83447265625]
RMSE:  43.564141500225546
r2:  0.673413397294574
RMSE :  43.564141500225546
r2:  0.673413397294574
time:  26.81170344352722

submission_scal_011104.csv
Epoch 421/10000
53/54 [============================>.] - ETA: 0s - loss: 41.4125 - mae: 4.3782 - mse: 41.4125Restoring model weights from the end of the best epoch: 121.
54/54 [==============================] - 0s 1ms/step - loss: 41.3760 - mae: 4.3769 - mse: 41.3760 - val_loss: 2043.6562 - val_mae: 30.7149 - val_mse: 2043.6562
Epoch 00421: early stopping
9/9 [==============================] - 0s 873us/step - loss: 1823.4142 - mae: 28.6813 - mse: 1823.4142
loss :  [1823.4141845703125, 28.681324005126953, 1823.4141845703125]
RMSE:  42.70145755880357
r2:  0.6862198662627312
RMSE :  42.70145755880357
r2:  0.6862198662627312
time:  32.07161784172058

: Dacon_ddarung 에서는 MinMaxScaler가 가장 좋게 나왔다.
"""

"""
<test_csv를 scaling 한 상태!>
1) scaling 안 했을 때
submission_ear_010904.csv : patience=300
Epoch 1583/10000
 1/54 [..............................] - ETA: 0s - loss: 731.1029 - mae: 20.2387 - mse: 731.1029Restoring model weights from the end of the best epoch: 1283.
54/54 [==============================] - 0s 1ms/step - loss: 680.0212 - mae: 16.9566 - mse: 680.0212 - val_loss: 2318.4683 - val_mae: 35.4183 - val_mse: 2318.4683
Epoch 01583: early stopping
9/9 [==============================] - 0s 620us/step - loss: 2078.1926 - mae: 32.7382 - mse: 2078.1926
loss :  [2078.192626953125, 32.73818588256836, 2078.192626953125]
RMSE:  45.58719855758717
r2:  0.6423766605731327

2) MinMaxScaler
submission_scal_011105.csv
Epoch 620/10000
51/54 [===========================>..] - ETA: 0s - loss: 93.6113 - mae: 6.6899 - mse: 93.6113  Restoring model weights from the end of the best epoch: 320.
54/54 [==============================] - 0s 1ms/step - loss: 92.1661 - mae: 6.6293 - mse: 92.1661 - val_loss: 1744.0785 - val_mae: 28.0958 - val_mse: 1744.0785
Epoch 00620: early stopping
9/9 [==============================] - 0s 619us/step - loss: 1918.5052 - mae: 29.3325 - mse: 1918.5052
loss :  [1918.5052490234375, 29.332548141479492, 1918.5052490234375]
RMSE:  43.80074457283419
r2:  0.6698562869285418
RMSE :  43.80074457283419

submission_scal_011106.csv
Epoch 532/10000
 1/54 [..............................] - ETA: 0s - loss: 79.0859 - mae: 6.8044 - mse: 79.0859Restoring model weights from the end of the best epoch: 232.
54/54 [==============================] - 0s 1ms/step - loss: 93.4613 - mae: 6.8592 - mse: 93.4613 - val_loss: 2093.0308 - val_mae: 31.0250 - val_mse: 2093.0308
Epoch 00532: early stopping
9/9 [==============================] - 0s 614us/step - loss: 1591.4371 - mae: 28.8819 - mse: 1591.4371
loss :  [1591.4371337890625, 28.881860733032227, 1591.4371337890625]
RMSE:  39.89282146471091
r2:  0.7261394022271431
time:  39.36497974395752

3) StandardScaler
submission_scal_011107.csv
Epoch 368/10000
 1/54 [..............................] - ETA: 0s - loss: 41.7058 - mae: 5.3628 - mse: 41.7058Restoring model weights from the end of the best epoch: 68.
54/54 [==============================] - 0s 1ms/step - loss: 52.9927 - mae: 5.3514 - mse: 52.9927 - val_loss: 2142.3677 - val_mae: 33.7541 - val_mse: 2142.3677
Epoch 00368: early stopping
9/9 [==============================] - 0s 696us/step - loss: 2031.6913 - mae: 31.7669 - mse: 2031.6913
loss :  [2031.6912841796875, 31.76691436767578, 2031.6912841796875]
RMSE:  45.0742842871601
r2:  0.6503788282142213
time:  26.738710641860962

submission_scal_011108.csv
Epoch 388/10000
51/54 [===========================>..] - ETA: 0s - loss: 46.7511 - mae: 4.6698 - mse: 46.7511Restoring model weights from the end of the best epoch: 88.
54/54 [==============================] - 0s 1ms/step - loss: 46.0786 - mae: 4.6497 - mse: 46.0786 - val_loss: 2197.6416 - val_mae: 32.9331 - val_mse: 2197.6416
Epoch 00388: early stopping
9/9 [==============================] - 0s 623us/step - loss: 1954.8584 - mae: 30.2947 - mse: 1954.8584
loss :  [1954.8583984375, 30.294689178466797, 1954.8583984375]
RMSE:  44.21377959390652
r2:  0.6636005098088609
time:  29.032681465148926

: Dacon_ddarung 에서는 MinMax 가 가장 좋게 나왔다.
"""