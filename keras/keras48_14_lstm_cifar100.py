import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, LSTM
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

x_train = x_train.reshape(50000, 96, 32)
x_test = x_test.reshape(10000, 96, 32)

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
model.add(LSTM(700, input_shape=(96, 32), activation='relu'))
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

# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")
# filepath = 'c:/study/_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True, verbose=1)
# mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1,
#                       filepath = filepath + 'LSTM_C100_' + date + '_' + filename)

import time
start = time.time()
model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=32, verbose=1)       # callbacks=[es, mcp]
end = time.time()

#4. evaluate, predict
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])

y_predict = model.predict(x_test)
print('time : ', end-start)

"""
Epoch 5/5
1250/1250 [==============================] - 176s 141ms/step - loss: 4.6058 - acc: 0.0094 - val_loss: 4.6077 - val_acc: 0.0083
313/313 [==============================] - 13s 42ms/step - loss: 4.6054 - acc: 0.0100
loss :  4.605445861816406
accuracy :  0.009999999776482582
time :  860.3885416984558
"""