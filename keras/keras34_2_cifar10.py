import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
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
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(32, 32, 3), activation='relu'))
model.add(Conv2D(64, (2,2)))
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
<1>
Epoch 9/100
1247/1250 [============================>.] - ETA: 0s - loss: 2.3028 - acc: 0.1000Restoring model weights from the end of the best epoch: 4.

Epoch 00009: val_loss did not improve from 2.30264
1250/1250 [==============================] - 8s 6ms/step - loss: 2.3028 - acc: 0.1000 - val_loss: 2.3029 - val_acc: 0.0952
Epoch 00009: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 2.3026 - acc: 0.1000
loss :  2.3025989532470703
accuracy :  0.10000000149011612
time :  71.3188259601593

<2>
Epoch 23/1000
1246/1250 [============================>.] - ETA: 0s - loss: 2.3028 - acc: 0.0993Restoring model weights from the end of the best epoch: 13.

Epoch 00023: val_loss did not improve from 2.30259
1250/1250 [==============================] - 17s 14ms/step - loss: 2.3028 - acc: 0.0992 - val_loss: 2.3027 - val_acc: 0.0980
Epoch 00023: early stopping
313/313 [==============================] - 3s 11ms/step - loss: 2.3026 - acc: 0.1000
loss :  2.3026247024536133
accuracy :  0.10000000149011612
time :  217.1584439277649

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
"""