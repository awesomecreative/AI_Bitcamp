import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

#1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) (60000, 28, 28)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(type(x_train), type(y_train)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, x_test.shape) # (60000, 28, 28, 1) (10000, 28, 28, 1)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))
# y의 class 개수가 10개 이므로 다중 분류이다.

#2. model
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2,2), input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(64, (2,2)))
model.add(Conv2D(64, (2,2)))    # (25, 25, 64)
model.add(Flatten())              # (40000, )
model.add(Dense(32, activation='relu'))     # input_shape => (batch_size, input_dim)=(60000, 40000)인데 행 무시 하므로 (40000,) 과 같다.
model.add(Dense(10, activation='softmax'))  

#3. compile, fit
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = 'c:/study/_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1,
                      filepath = filepath + 'CNN_1_' + date + '_' + filename)

import time
start = time.time()
model.fit(x_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1,
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
Epoch 100/100
1875/1875 [==============================] - 10s 5ms/step - loss: 0.0585 - acc: 0.9929
313/313 [==============================] - 1s 3ms/step - loss: 1.3023 - acc: 0.9602
loss :  1.3022733926773071
accuracy :  0.9602000117301941

<2>
Epoch 11/100
1497/1500 [============================>.] - ETA: 0s - loss: 0.0453 - acc: 0.9871Restoring model weights from the end of the best epoch: 6.
Epoch 00011: val_loss did not improve from 0.11931
1500/1500 [==============================] - 8s 6ms/step - loss: 0.0454 - acc: 0.9871 - val_loss: 0.1740 - val_acc: 0.9654
Epoch 00011: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.1190 - acc: 0.9684
loss :  0.11896849423646927
accuracy :  0.9684000015258789
"""

"""
CNN에 넣으려면 독립변수 데이터를 4차원 텐서로 reshape 해야한다.
reshape 할 때는 원하는 모양을 적으면 된다.
스칼라 (3, )를 벡터 (3,1)로 바꿀 때에는 .reshape(3,1) 또는 .reshape(-1,1)로 표현 가능했지만,
텐서로 바꿀 때에는 -1이 안 통한다. 그냥 다 적어주는 게 좋다.

one-hot 안 해줬으므로 categorical이 아니라 sparse_categorical 쓴다.

Dense일 때는 CPU 쓰면 더 좋았지만 CNN일 때는 GPU 쓰는 게 더 빠르다.
"""