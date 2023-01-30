import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

#1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) (60000, 28, 28)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))
# y의 class 개수가 10개 이므로 다중 분류이다.

#2. model
model = Sequential()
model.add(Conv1D(128, 2, input_shape=(28, 28)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))     # input_shape => (batch_size, input_dim)=(60000, 40000)인데 행 무시 하므로 (40000,) 과 같다.
model.add(Dense(10, activation='softmax'))  

#3. compile, fit
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")
# filepath = 'c:/study/_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
# mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1,
#                       filepath = filepath + 'LSTM_1_' + date + '_' + filename)

import time
start = time.time()
model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=1) # callbacks=[es,mcp]
end = time.time()

#4. evaluate, predict
results = model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])

y_predict = model.predict(x_test)

print('time : ', end-start)

"""
Epoch 10/10
1500/1500 [==============================] - 5s 3ms/step - loss: 0.1971 - acc: 0.9489 - val_loss: 0.1955 - val_acc: 0.9514
313/313 [==============================] - 1s 3ms/step - loss: 0.1954 - acc: 0.9520
loss :  0.19543041288852692
accuracy :  0.9520000219345093
time :  54.432358264923096
"""