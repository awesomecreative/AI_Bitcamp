import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

#1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train/255.
x_test = x_test/255.

print(np.unique(y_train, return_counts=True)) 
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 
#        51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 
#        68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 
#        85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]), array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
#        500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int64))
# y의 class 개수는 100개로 다중분류이다.

#2. model
model = Sequential()
model.add(Dense(700, input_shape=(32*32*3, ), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(350, activation='relu'))
model.add(Dense(256, activation='linear'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))  

#3. compile, fit
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = 'c:/study/_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1,
                      filepath = filepath + 'DNN_C100_' + date + '_' + filename)

import time
start = time.time()
model.fit(x_train, y_train, validation_split=0.2, epochs=1000, batch_size=32, verbose=1,
          callbacks=[es, mcp])
end = time.time()

#4. evaluate, predict
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])

y_predict = model.predict(x_test)
print('time : ', end-start)

"""
★ CNN 결과
<1>
Epoch 27/1000
1248/1250 [============================>.] - ETA: 0s - loss: 4.6056 - acc: 0.0090Restoring model weights from the end of the best epoch: 17.

Epoch 00027: val_loss did not improve from 3.71098
1250/1250 [==============================] - 34s 27ms/step - loss: 4.6056 - acc: 0.0090 - val_loss: 4.6080 - val_acc: 0.0077
Epoch 00027: early stopping
313/313 [==============================] - 3s 10ms/step - loss: 3.7035 - acc: 0.1247
loss :  3.7035093307495117
accuracy :  0.12470000237226486
time :  904.4051067829132

★ DNN 결과
<1>
1250/1250 [==============================] - 6s 5ms/step - loss: 4.2428 - acc: 0.0415 - val_loss: 4.3485 - val_acc: 0.0322
Epoch 29/1000
1244/1250 [============================>.] - ETA: 0s - loss: 4.2448 - acc: 0.0425Restoring model weights from the end of the best epoch: 19.

Epoch 00029: val_loss did not improve from 4.20848
1250/1250 [==============================] - 6s 5ms/step - loss: 4.2447 - acc: 0.0425 - val_loss: 4.2107 - val_acc: 0.0495
Epoch 00029: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 4.1978 - acc: 0.0502
loss :  4.197827339172363
accuracy :  0.050200000405311584
time :  184.52982306480408

<2> : Scaler 추가
Epoch 68/1000
1249/1250 [============================>.] - ETA: 0s - loss: 3.5473 - acc: 0.1472Restoring model weights from the end of the best epoch: 58.

Epoch 00068: val_loss did not improve from 3.54498
1250/1250 [==============================] - 7s 5ms/step - loss: 3.5472 - acc: 0.1473 - val_loss: 3.6187 - val_acc: 0.1496
Epoch 00068: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 3.5427 - acc: 0.1657
loss :  3.5426764488220215
accuracy :  0.1657000035047531
time :  442.9080743789673

★ 결론
DNN이 성능이 더 좋게 나왔다.
"""