import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,) (60000, 28, 28)
print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)

# <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train), type(y_train))

# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train/255. # scaling 작업
x_test = x_test/255. # scaling 작업

print(x_train.shape, x_test.shape)  #

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))
# y의 class 개수가 10개 이므로 다중 분류이다.

# 2. model
model = Sequential()            # input_shape=(28*28, )로 적어도 괜찮다.
model.add(Dense(256, input_shape=(784,), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='linear'))
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. compile, fit
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['acc'])

date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
filepath = 'c:/study/_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

es = EarlyStopping(monitor='val_loss', mode='min', patience=5,
                   restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='min', save_best_only=True, verbose=1,
                      filepath=filepath + 'DNN_1_' + date + '_' + filename)

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
★ CNN 결과
Epoch 13/100
1500/1500 [==============================] - ETA: 0s - loss: 0.0572 - acc: 0.9829Restoring model weights from the end of the best epoch: 8.

Epoch 00013: val_loss did not improve from 0.10090
1500/1500 [==============================] - 7s 5ms/step - loss: 0.0572 - acc: 0.9829 - val_loss: 0.1323 - val_acc: 0.9687
Epoch 00013: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.0914 - acc: 0.9740
loss :  0.09140654653310776
accuracy :  0.9739999771118164
time :  93.93943643569946

★ DNN 결과
<1> : first layer dense(500)
Epoch 18/100
1500/1500 [==============================] - ETA: 0s - loss: 0.1700 - acc: 0.9482Restoring model weights from the end of the best epoch: 13.

Epoch 00018: val_loss did not improve from 0.11838
1500/1500 [==============================] - 5s 4ms/step - loss: 0.1700 - acc: 0.9482 - val_loss: 0.1241 - val_acc: 0.9658
Epoch 00018: early stopping
313/313 [==============================] - 1s 2ms/step - loss: 0.1247 - acc: 0.9625
loss :  0.12465877830982208
accuracy :  0.9624999761581421
time :  97.30463242530823

<2> : first layer dense(256)
Epoch 13/100
1493/1500 [============================>.] - ETA: 0s - loss: 0.0589 - acc: 0.9819Restoring model weights from the end of the best epoch: 8.

Epoch 00013: val_loss did not improve from 0.07987
1500/1500 [==============================] - 5s 4ms/step - loss: 0.0589 - acc: 0.9819 - val_loss: 0.0888 - val_acc: 0.9794
Epoch 00013: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.0777 - acc: 0.9791
loss :  0.0777161717414856
accuracy :  0.9790999889373779
time :  72.44193768501282

<3> : <2>와 동일
Epoch 18/100
1490/1500 [============================>.] - ETA: 0s - loss: 0.0501 - acc: 0.9845Restoring model weights from the end of the best epoch: 13.

Epoch 00018: val_loss did not improve from 0.08382
1500/1500 [==============================] - 5s 4ms/step - loss: 0.0504 - acc: 0.9844 - val_loss: 0.0883 - val_acc: 0.9783
Epoch 00018: early stopping
313/313 [==============================] - 1s 2ms/step - loss: 0.0800 - acc: 0.9800
loss :  0.07999800890684128
accuracy :  0.9800000190734863
time :  97.11717247962952

<4> : MinMaxScaler 추가
Epoch 38/100
1489/1500 [============================>.] - ETA: 0s - loss: 0.0577 - acc: 0.9814Restoring model weights from the end of the best epoch: 33.

Epoch 00038: val_loss did not improve from 0.08233
1500/1500 [==============================] - 5s 3ms/step - loss: 0.0576 - acc: 0.9815 - val_loss: 0.0931 - val_acc: 0.9783
Epoch 00038: early stopping
313/313 [==============================] - 1s 2ms/step - loss: 0.0777 - acc: 0.9784
loss :  0.0776500329375267
accuracy :  0.9783999919891357
time :  200.86281490325928

★ 결론
DNN의 성능이 더 좋다.
"""

"""
<학습내용>
DNN은 Conv2D를 사용하지않고 reshape, scaling만 하고 Dense로 바로 모델 구성하면 된다.
"""