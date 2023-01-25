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


x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train/255. # Min, Max Scaler
x_test = x_test/255.

print(np.unique(y_train, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))
# y의 class 개수는 10개로 다중분류이다.

#2. model
model = Sequential()
model.add(Dense(512, input_shape=(32*32*3, ), activation='relu',))
model.add(Dropout(0.3))
model.add(Dense(256, activation='linear'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
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
                      filepath = filepath + 'DNN_C10_' + date + '_' + filename)

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

★ DNN 결과
<1>
Epoch 27/1000
1243/1250 [============================>.] - ETA: 0s - loss: 1.8990 - acc: 0.2905Restoring model weights from the end of the best epoch: 17.

Epoch 00027: val_loss did not improve from 1.89276
1250/1250 [==============================] - 5s 4ms/step - loss: 1.8987 - acc: 0.2907 - val_loss: 1.9375 - val_acc: 0.2927
Epoch 00027: early stopping
313/313 [==============================] - 1s 2ms/step - loss: 1.8787 - acc: 0.3066
loss :  1.8787165880203247
accuracy :  0.30660000443458557
time :  135.38686561584473

<2>
Epoch 27/1000
1243/1250 [============================>.] - ETA: 0s - loss: 1.8370 - acc: 0.3210Restoring model weights from the end of the best epoch: 17.

Epoch 00027: val_loss did not improve from 1.82583
1250/1250 [==============================] - 5s 4ms/step - loss: 1.8364 - acc: 0.3209 - val_loss: 1.8411 - val_acc: 0.3466
Epoch 00027: early stopping
313/313 [==============================] - 1s 2ms/step - loss: 1.8123 - acc: 0.3593
loss :  1.8122718334197998
accuracy :  0.35929998755455017
time :  136.71368265151978

<3>
Epoch 39/1000
1244/1250 [============================>.] - ETA: 0s - loss: 1.8127 - acc: 0.3364Restoring model weights from the end of the best epoch: 29.

Epoch 00039: val_loss did not improve from 1.77405
1250/1250 [==============================] - 5s 4ms/step - loss: 1.8128 - acc: 0.3362 - val_loss: 1.8136 - val_acc: 0.3539
Epoch 00039: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 1.7626 - acc: 0.3753
loss :  1.762632966041565
accuracy :  0.37529999017715454
time :  207.22048211097717

<4> : Scaler 추가
Epoch 63/1000
1243/1250 [============================>.] - ETA: 0s - loss: 1.3753 - acc: 0.5060Restoring model weights from the end of the best epoch: 53.

Epoch 00063: val_loss did not improve from 1.39490
1250/1250 [==============================] - 5s 4ms/step - loss: 1.3753 - acc: 0.5058 - val_loss: 1.3984 - val_acc: 0.5051
Epoch 00063: early stopping
313/313 [==============================] - 1s 2ms/step - loss: 1.3898 - acc: 0.5026
loss :  1.3898097276687622
accuracy :  0.5026000142097473
time :  344.63273549079895

★ 결론
cifar10 에서는 CNN이 더 좋은 것 같다.
"""