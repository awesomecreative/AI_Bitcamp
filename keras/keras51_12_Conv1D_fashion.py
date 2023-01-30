from tensorflow.keras.datasets import fashion_mnist

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, LSTM, Conv1D


#1. data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) (60000, 28, 28)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

print(x_train.shape, x_test.shape)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))
# y의 class 개수가 10개 이므로 다중 분류이다.

# 2. model
model = Sequential()
model.add(Conv1D(512, 2, input_shape=(28, 28)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. compile, fit
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['acc'])

import datetime, time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")
# filepath = 'c:/study/_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# es = EarlyStopping(monitor='val_loss', mode='min', patience=5,
#                    restore_best_weights=True, verbose=1)
# mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1,
#                       filepath=filepath + 'LSTM_F_' + date + '_' + filename)

start = time.time()
model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=1) # callbacks=[es, mcp]
end = time.time()

# 4. evaluate, predict
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])

y_predict = model.predict(x_test)

print('time : ', end-start)

#5. ex, mcp 적용 / val 적용

"""
Epoch 10/10
1500/1500 [==============================] - 7s 4ms/step - loss: 0.2677 - acc: 0.9003 - val_loss: 0.3295 - val_acc: 0.8821
313/313 [==============================] - 1s 3ms/step - loss: 0.3617 - acc: 0.8713
loss :  0.3616642951965332
accuracy :  0.8712999820709229
time :  68.2035961151123
"""