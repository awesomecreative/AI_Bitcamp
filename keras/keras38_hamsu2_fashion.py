from tensorflow.keras.datasets import fashion_mnist

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D


#1. data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) (60000, 28, 28)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, x_test.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))
# y의 class 개수가 10개 이므로 다중 분류이다.

# # 2. model
# model = Sequential()
# model.add(Conv2D(filters=128, kernel_size=(2, 2), input_shape=(28, 28, 1),
#                  padding='same',
#                  activation='relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(64, (2, 2), padding='same'))
# model.add(Conv2D(64, (2, 2), padding='same'))    # (25, 25, 64)
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(10, activation='softmax'))

input1 = Input(shape=(28,28,1))
dense1 = Conv2D(filters=128, kernel_size=(2,2), padding='same', activation='relu')(input1)
dense2 = MaxPooling2D()(dense1)
dense3 = Conv2D(64, (2,2), padding='same')(dense2)
dense4 = Conv2D(64, (2,2), padding='same')(dense3)
dense5 = Flatten()(dense4)
dense6 = Dense(128, activation='relu')(dense5)
dense7 = Dense(64, activation='relu')(dense6)
dense8 = Dense(32, activation='relu')(dense7)
output1 = Dense(10, activation='softmax')(dense8)
model = Model(inputs=input1, outputs=output1)
model.summary()

# 3. compile, fit
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['acc'])

import datetime, time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = 'c:/study/_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', mode='min', patience=5,
                   restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1,
                      filepath=filepath + 'CNN_FH_' + date + '_' + filename)

start = time.time()
model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1,
          callbacks=[es, mcp])
end = time.time()

# 4. evaluate, predict
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])

y_predict = model.predict(x_test)

print('time : ', end-start)

#5. ex, mcp 적용 / val 적용

"""
Epoch 12/100
1498/1500 [============================>.] - ETA: 0s - loss: 0.1870 - acc: 0.9337Restoring model weights from the end of the best epoch: 7.

Epoch 00012: val_loss did not improve from 0.29890
1500/1500 [==============================] - 9s 6ms/step - loss: 0.1872 - acc: 0.9336 - val_loss: 0.3394 - val_acc: 0.8987
Epoch 00012: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.3124 - acc: 0.8964
loss :  0.31235241889953613
accuracy :  0.896399974822998
time :  116.04207134246826
"""