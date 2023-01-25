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

print(x_train.shape, x_test.shape)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000], dtype=int64))
# y의 class 개수가 10개 이므로 다중 분류이다.

# 2. model
model = Sequential()
model.add(Dense(500, input_shape=(28*28, ), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
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
                      filepath=filepath + 'DNN_F_' + date + '_' + filename)

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

★ DNN 결과
<1> : Dropout 0.3 0.3
Epoch 27/100
1493/1500 [============================>.] - ETA: 0s - loss: 0.2643 - acc: 0.9016Restoring model weights from the end of the best epoch: 22.

Epoch 00027: val_loss did not improve from 0.30679
1500/1500 [==============================] - 6s 4ms/step - loss: 0.2647 - acc: 0.9015 - val_loss: 0.3196 - val_acc: 0.8910
Epoch 00027: early stopping
313/313 [==============================] - 1s 2ms/step - loss: 0.3283 - acc: 0.8829
loss :  0.32825708389282227
accuracy :  0.8828999996185303
time :  161.96592140197754

<2> : Dropout 0.3 0.2
Epoch 30/100
1493/1500 [============================>.] - ETA: 0s - loss: 0.2497 - acc: 0.9074Restoring model weights from the end of the best epoch: 25.

Epoch 00030: val_loss did not improve from 0.29405
1500/1500 [==============================] - 6s 4ms/step - loss: 0.2501 - acc: 0.9074 - val_loss: 0.3039 - val_acc: 0.8940
Epoch 00030: early stopping
313/313 [==============================] - 1s 2ms/step - loss: 0.3253 - acc: 0.8858
loss :  0.32526034116744995
accuracy :  0.8858000040054321
time :  180.90508151054382

<3> : first layer dense 500 Dropout 0.3 0.2
Epoch 27/100
1499/1500 [============================>.] - ETA: 0s - loss: 0.2435 - acc: 0.9097Restoring model weights from the end of the best epoch: 22.

Epoch 00027: val_loss did not improve from 0.29998
1500/1500 [==============================] - 6s 4ms/step - loss: 0.2435 - acc: 0.9097 - val_loss: 0.3096 - val_acc: 0.8902
Epoch 00027: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.3377 - acc: 0.8836
loss :  0.33770912885665894
accuracy :  0.8835999965667725
time :  164.93997764587402

<4> : <3> + second layer dense 256
Epoch 23/100
1492/1500 [============================>.] - ETA: 0s - loss: 0.2502 - acc: 0.9050Restoring model weights from the end of the best epoch: 18.

Epoch 00023: val_loss did not improve from 0.29926
1500/1500 [==============================] - 6s 4ms/step - loss: 0.2503 - acc: 0.9050 - val_loss: 0.3066 - val_acc: 0.8894
Epoch 00023: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.3252 - acc: 0.8857
loss :  0.3251535892486572
accuracy :  0.885699987411499
time :  139.59137725830078

<5> : scaler 추가
Epoch 19/100
1492/1500 [============================>.] - ETA: 0s - loss: 0.2964 - acc: 0.8890Restoring model weights from the end of the best epoch: 14.

Epoch 00019: val_loss did not improve from 0.31862
1500/1500 [==============================] - 6s 4ms/step - loss: 0.2967 - acc: 0.8890 - val_loss: 0.3190 - val_acc: 0.8852
Epoch 00019: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.3464 - acc: 0.8757
loss :  0.3464363217353821
accuracy :  0.8756999969482422
time :  119.97824954986572

★ 결론
CNN과 DNN 거의 비슷하다.
"""