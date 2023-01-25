import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

#1. data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))
# y의 class 개수는 10개로 다중분류이다.

#2. model
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(32, 32, 3), padding='same', activation='relu',))
model.add(Conv2D(64, (2,2), padding='same'))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))  

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
                      filepath = filepath + 'CNN_2_' + date + '_' + filename)

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
<3> : Dense 128-64-32-10 으로 모델 바꾸니까 정확도 올라감!!
Epoch 18/1000
1249/1250 [============================>.] - ETA: 0s - loss: 0.3986 - acc: 0.8637Restoring model weights from the end of the best epoch: 8.

Epoch 00018: val_loss did not improve from 1.34101
1250/1250 [==============================] - 9s 7ms/step - loss: 0.3985 - acc: 0.8637 - val_loss: 2.1495 - val_acc: 0.5300
Epoch 00018: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 1.3506 - acc: 0.5490
loss :  1.3506280183792114
accuracy :  0.5490000247955322
time :  210.5506010055542

Padding & MaxPooling2D 후...
<1>
Epoch 16/1000
1248/1250 [============================>.] - ETA: 0s - loss: 0.2335 - acc: 0.9222Restoring model weights from the end of the best epoch: 6.

Epoch 00016: val_loss did not improve from 1.10348
1250/1250 [==============================] - 11s 9ms/step - loss: 0.2334 - acc: 0.9222 - val_loss: 1.8537 - val_acc: 0.6342
Epoch 00016: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 1.1205 - acc: 0.6331
loss :  1.1205172538757324
accuracy :  0.6330999732017517
time :  171.81443238258362

<2>
Epoch 15/1000
1244/1250 [============================>.] - ETA: 0s - loss: 0.3054 - acc: 0.8990Restoring model weights from the end of the best epoch: 5.

Epoch 00015: val_loss did not improve from 1.10714
1250/1250 [==============================] - 10s 8ms/step - loss: 0.3055 - acc: 0.8989 - val_loss: 1.5631 - val_acc: 0.6339
Epoch 00015: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 1.1224 - acc: 0.6210
loss :  1.1224157810211182
accuracy :  0.6209999918937683
time :  158.20213389396667

<3>
Epoch 17/1000
1249/1250 [============================>.] - ETA: 0s - loss: 0.2920 - acc: 0.9080Restoring model weights from the end of the best epoch: 7.

Epoch 00017: val_loss did not improve from 1.07199
1250/1250 [==============================] - 11s 9ms/step - loss: 0.2922 - acc: 0.9079 - val_loss: 1.6568 - val_acc: 0.6374
Epoch 00017: early stopping
313/313 [==============================] - 1s 4ms/step - loss: 1.0810 - acc: 0.6431
loss :  1.0810163021087646
accuracy :  0.6431000232696533
time :  180.1789333820343

: 성능이 더 좋아졌다.
"""