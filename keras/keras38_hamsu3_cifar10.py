import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

#1. data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))
# y의 class 개수는 10개로 다중분류이다.

# #2. model
# model = Sequential()
# model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(32, 32, 3), padding='same', activation='relu',))
# model.add(Conv2D(64, (2,2), padding='same'))
# model.add(MaxPooling2D())
# model.add(Conv2D(32, (3,3)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))

input1 = Input(shape=(32,32,3))
dense1 = Conv2D(filters=128, kernel_size=(2,2), padding='same', activation='relu')(input1)
dense2 = Conv2D(64, (2,2), padding='same')(dense1)
dense3 = MaxPooling2D()(dense2)
dense4 = Conv2D(32, (3,3))(dense3)
dense5 = Flatten()(dense4)
dense6 = Dense(128, activation='relu')(dense5)
dense7 = Dense(64, activation='relu')(dense6)
dense8 = Dense(32, activation='relu')(dense7)
output1 = Dense(10, activation='softmax')(dense8)
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
                      filepath = filepath + 'CNN_C10H_' + date + '_' + filename)

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
Epoch 17/1000
1245/1250 [============================>.] - ETA: 0s - loss: 0.2149 - acc: 0.9294Restoring model weights from the end of the best epoch: 7.

Epoch 00017: val_loss did not improve from 1.06567
1250/1250 [==============================] - 11s 9ms/step - loss: 0.2149 - acc: 0.9294 - val_loss: 1.8086 - val_acc: 0.6250
Epoch 00017: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 1.0610 - acc: 0.6468
loss :  1.0610127449035645
accuracy :  0.6467999815940857
time :  185.27032375335693
"""