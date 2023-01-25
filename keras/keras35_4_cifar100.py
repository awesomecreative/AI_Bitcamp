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
model.add(Conv2D(filters=512, kernel_size=(3,3), input_shape=(32, 32, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3,3), padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(128, (3,3), padding='same'))
model.add(Conv2D(64, (3,3), padding='same'))
model.add(Conv2D(32, (3,3), padding='same'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
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
                      filepath = filepath + 'CNN_3_' + date + '_' + filename)

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
Padding & MaxPooling2D 전...
<3>
Epoch 13/1000
1248/1250 [============================>.] - ETA: 0s - loss: 6.1029 - acc: 0.0095Restoring model weights from the end of the best epoch: 3.

Epoch 00013: val_loss did not improve from 4.60269
1250/1250 [==============================] - 30s 24ms/step - loss: 6.1005 - acc: 0.0095 - val_loss: 4.6079 - val_acc: 0.0077
Epoch 00013: early stopping
313/313 [==============================] - 2s 8ms/step - loss: 4.6011 - acc: 0.0117
loss :  4.601058006286621
accuracy :  0.011699999682605267
time :  392.4617533683777

Padding & MaxPooling2D 후...
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

<2>
Epoch 15/1000
1249/1250 [============================>.] - ETA: 0s - loss: 7.0340 - acc: 0.0100Restoring model weights from the end of the best epoch: 5.

Epoch 00015: val_loss did not improve from 3.79078
1250/1250 [==============================] - 33s 27ms/step - loss: 7.0321 - acc: 0.0100 - val_loss: 4.6080 - val_acc: 0.0077
Epoch 00015: early stopping
313/313 [==============================] - 3s 10ms/step - loss: 3.7773 - acc: 0.1209
loss :  3.7772586345672607
accuracy :  0.120899997651577
time :  494.2467739582062

<3>
Epoch 16/1000
1250/1250 [==============================] - ETA: 0s - loss: 4.6078 - acc: 0.0097Restoring model weights from the end of the best epoch: 6.

Epoch 00016: val_loss did not improve from 3.88056
1250/1250 [==============================] - 34s 27ms/step - loss: 4.6078 - acc: 0.0097 - val_loss: 4.6079 - val_acc: 0.0077
Epoch 00016: early stopping
313/313 [==============================] - 3s 10ms/step - loss: 3.8706 - acc: 0.1042
loss :  3.870574712753296
accuracy :  0.10419999808073044
time :  551.0217361450195

: 성능이 많이 좋아졌다.
"""