import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D
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

# #2. model
# model = Sequential()
# model.add(Conv2D(filters=512, kernel_size=(3,3), input_shape=(32, 32, 3), padding='same', activation='relu'))
# model.add(Conv2D(256, (3,3), padding='same'))
# model.add(MaxPooling2D())
# model.add(Conv2D(128, (3,3), padding='same'))
# model.add(Conv2D(64, (3,3), padding='same'))
# model.add(Conv2D(32, (3,3), padding='same'))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(350, activation='relu'))
# model.add(Dense(256, activation='linear'))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(100, activation='softmax'))

input1 = Input(shape=(32,32,3))
dense1 = Conv2D(filters=512, kernel_size=(3,3), padding='same', activation='relu')(input1)
dense2 = Conv2D(256, (3,3), padding='same')(dense1)
dense3 = MaxPooling2D()(dense2)
dense4 = Conv2D(128, (3,3), padding='same')(dense3)
dense5 = Flatten()(dense4)
dense6 = Dense(512, activation='relu')(dense5)
dense7 = Dropout(0.5)(dense6)
dense8 = Dense(350, activation='relu')(dense7)
dense9 = Dense(256, activation='linear')(dense8)
dense10 = Dropout(0.5)(dense9)
dense11 = Dense(128, activation='relu')(dense10)
output1 = Dense(100, activation='softmax')(dense11)
model = Model(inputs=input1, outputs=output1)
model.summary()

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
                      filepath = filepath + 'CNN_C100H_' + date + '_' + filename)

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
Epoch 19/1000
1249/1250 [============================>.] - ETA: 0s - loss: 3.9197 - acc: 0.0925Restoring model weights from the end of the best epoch: 9.

Epoch 00019: val_loss did not improve from 3.74079
1250/1250 [==============================] - 35s 28ms/step - loss: 3.9194 - acc: 0.0926 - val_loss: 3.8473 - val_acc: 0.1040
Epoch 00019: early stopping
313/313 [==============================] - 3s 10ms/step - loss: 3.7281 - acc: 0.1264
loss :  3.72806715965271
accuracy :  0.12639999389648438
time :  669.6230418682098
"""