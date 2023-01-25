from tensorflow.keras.datasets import fashion_mnist

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D


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

# 2. model
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2), input_shape=(28, 28, 1),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2, 2), padding='same'))
model.add(Conv2D(64, (2, 2), padding='same'))    # (25, 25, 64)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

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
                      filepath=filepath + 'CNN_F_' + date + '_' + filename)

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
결과
<1>
1500/1500 [==============================] - 7s 5ms/step - loss: 2.3028 - acc: 0.0973 - val_loss: 2.3027 - val_acc: 0.0957
Epoch 10/100
1498/1500 [============================>.] - ETA: 0s - loss: 2.3028 - acc: 0.1008Restoring model weights from the end of the best epoch: 5.

Epoch 00010: val_loss did not improve from 2.30265
1500/1500 [==============================] - 7s 5ms/step - loss: 2.3028 - acc: 0.1007 - val_loss: 2.3027 - val_acc: 0.0983
Epoch 00010: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 2.3026 - acc: 0.1000
loss :  2.302640199661255
accuracy :  0.10000000149011612
time :  74.53036499023438

<2>
Epoch 11/100
1498/1500 [============================>.] - ETA: 0s - loss: 0.1796 - acc: 0.9361Restoring model weights from the end of the best epoch: 6.

Epoch 00011: val_loss did not improve from 0.30354
1500/1500 [==============================] - 8s 6ms/step - loss: 0.1796 - acc: 0.9361 - val_loss: 0.3142 - val_acc: 0.9024
Epoch 00011: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.3109 - acc: 0.8934
loss :  0.3108886778354645
accuracy :  0.8934000134468079
time :  94.5241916179657

<3>
Epoch 11/100
1496/1500 [============================>.] - ETA: 0s - loss: 0.1803 - acc: 0.9355Restoring model weights from the end of the best epoch: 6.

Epoch 00011: val_loss did not improve from 0.29920
1500/1500 [==============================] - 8s 6ms/step - loss: 0.1802 - acc: 0.9355 - val_loss: 0.3277 - val_acc: 0.8982
Epoch 00011: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.3204 - acc: 0.8952
loss :  0.3203740119934082
accuracy :  0.8952000141143799
time :  94.23063492774963
"""