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
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1, restore_best_weights=True)

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

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('ddarung loss')
plt.legend(loc='center right')
plt.show()

y_submit = model.predict(test_csv)

submission['count'] = y_submit
submission.to_csv(path + 'submission_ear_010908.csv')
print ('RMSE : ', rmse)
print ('r2: ', r2)
print('time: ', end-start)

"""
submission_ear_010901.csv
Epoch 42/1000
 1/54 [..............................] - ETA: 0s - loss: 4277.9771 - mae: 45.5168 - mse: 4277.9771Restoring model weights from the end of the best epoch: 37.
54/54 [==============================] - 0s 1ms/step - loss: 2942.1389 - mae: 40.5740 - mse: 2942.1389 - val_loss: 3124.9749 - val_mae: 44.3163 - val_mse: 3124.9749
Epoch 00042: early stopping
9/9 [==============================] - 0s 679us/step - loss: 2371.5562 - mae: 34.9637 - mse: 2371.5562
loss :  [2371.55615234375, 34.96373748779297, 2371.55615234375]
RMSE:  48.69862452690995
r2:  0.5918935723958523

submission_ear_010902.csv : patience=50
Epoch 112/1000
 1/54 [..............................] - ETA: 0s - loss: 837.8738 - mae: 24.4316 - mse: 837.8738Restoring model weights from the end of the best epoch: 62.
54/54 [==============================] - 0s 1ms/step - loss: 2810.7090 - mae: 39.3477 - mse: 2810.7090 - val_loss: 3049.5598 - val_mae: 41.9973 - val_mse: 3049.5598
Epoch 00112: early stopping
9/9 [==============================] - 0s 639us/step - loss: 2362.3618 - mae: 35.2341 - mse: 2362.3618
loss :  [2362.36181640625, 35.23408508300781, 2362.36181640625]
RMSE:  48.604134110463335
r2:  0.59347574172481

submission_ear_010903.csv : patience=100
Epoch 270/10000
 1/54 [..............................] - ETA: 0s - loss: 4067.7026 - mae: 52.0985 - mse: 4067.7026Restoring model weights from the end of the best epoch: 170.
54/54 [==============================] - 0s 1ms/step - loss: 2601.6960 - mae: 37.0558 - mse: 2601.6960 - val_loss: 2963.4443 - val_mae: 41.2382 - val_mse: 2963.4443
Epoch 00270: early stopping
9/9 [==============================] - 0s 641us/step - loss: 2417.9609 - mae: 36.1570 - mse: 2417.9609
loss :  [2417.9609375, 36.157047271728516, 2417.9609375]
RMSE:  49.17276874470104
r2:  0.5839079958934035

submission_ear_010904.csv : patience=300
Epoch 1583/10000
 1/54 [..............................] - ETA: 0s - loss: 731.1029 - mae: 20.2387 - mse: 731.1029Restoring model weights from the end of the best epoch: 1283.
54/54 [==============================] - 0s 1ms/step - loss: 680.0212 - mae: 16.9566 - mse: 680.0212 - val_loss: 2318.4683 - val_mae: 35.4183 - val_mse: 2318.4683
Epoch 01583: early stopping
9/9 [==============================] - 0s 620us/step - loss: 2078.1926 - mae: 32.7382 - mse: 2078.1926
loss :  [2078.192626953125, 32.73818588256836, 2078.192626953125]
RMSE:  45.58719855758717
r2:  0.6423766605731327

submission_ear_010905.csv : patience=500
Epoch 1967/10000
 1/54 [..............................] - ETA: 0s - loss: 1064.1331 - mae: 24.1947 - mse: 1064.1331Restoring model weights from the end of the best epoch: 1467.
54/54 [==============================] - 0s 1ms/step - loss: 848.3273 - mae: 20.4259 - mse: 848.3273 - val_loss: 2682.2568 - val_mae: 37.1551 - val_mse: 2682.2568
Epoch 01967: early stopping
9/9 [==============================] - 0s 601us/step - loss: 2230.1165 - mae: 33.0333 - mse: 2230.1165
loss :  [2230.116455078125, 33.03329849243164, 2230.116455078125]
RMSE:  47.22410962875531
r2:  0.6162330223416199

submission_ear_010906.csv : patience=500 : 20 50 80 100 60 40 10 1
Epoch 1809/10000
 1/54 [..............................] - ETA: 0s - loss: 301.7531 - mae: 12.1156 - mse: 301.7531Restoring model weights from the end of the best epoch: 1309.
54/54 [==============================] - 0s 1ms/step - loss: 312.7692 - mae: 11.2589 - mse: 312.7692 - val_loss: 2732.9297 - val_mae: 36.7729 - val_mse: 2732.9297
Epoch 01809: early stopping
9/9 [==============================] - 0s 619us/step - loss: 2089.8171 - mae: 31.2277 - mse: 2089.8171
loss :  [2089.817138671875, 31.22768211364746, 2089.817138671875]
RMSE:  45.714517624272275
r2:  0.6403762809019855

submission_ear_010907.csv : patience=300 : 20 50 80 100 60 40 10 1
Epoch 5441/10000
54/54 [==============================] - ETA: 0s - loss: 7002.1348 - mae: 66.4450 - mse: 7002.1348Restoring model weights from the end of the best epoch: 5141.
54/54 [==============================] - 0s 1ms/step - loss: 7002.1348 - mae: 66.4450 - mse: 7002.1348 - val_loss: 7542.4932 - val_mae: 68.7031 - val_mse: 7542.4932
Epoch 05441: early stopping
9/9 [==============================] - 0s 586us/step - loss: 5913.2686 - mae: 61.6877 - mse: 5913.2686
loss :  [5913.2685546875, 61.68766784667969, 5913.2685546875]
RMSE:  76.89778135994511
r2:  -0.017577896170942076

submission_ear_010908.csv
Epoch 160/10000
 1/54 [..............................] - ETA: 0s - loss: 3110.8145 - mae: 46.5735 - mse: 3110.8145Restoring model weights from the end of the best epoch: 60.
54/54 [==============================] - 0s 1ms/step - loss: 2834.3323 - mae: 39.3280 - mse: 2834.3323 - val_loss: 3272.6370 - val_mae: 42.5423 - val_mse: 3272.6370
Epoch 00160: early stopping
9/9 [==============================] - 0s 619us/step - loss: 2319.9993 - mae: 35.3145 - mse: 2319.9993
loss :  [2319.999267578125, 35.31449890136719, 2319.999267578125]
RMSE:  48.166371271152514
r2:  0.6007656480338393
"""