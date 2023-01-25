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

# 1. data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)  # (60000, 28, 28) (60000,) (60000, 28, 28)
print(x_test.shape, y_test.shape)  # (10000, 28, 28) (10000,)

# <class 'numpy.ndarray'> <class 'numpy.ndarray'>
print(type(x_train), type(y_train))

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(x_train.shape, x_test.shape)  # (60000, 28, 28, 1) (10000, 28, 28, 1)

print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))
# y의 class 개수가 10개 이므로 다중 분류이다.

# 2. model
model = Sequential()
model.add(Conv2D(filters=128, kernel_size=(2, 2), input_shape=(28, 28, 1),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2, 2), padding='same'))
model.add(Conv2D(64, (2, 2), padding='same'))    # (25, 25, 64)
model.add(Flatten())              # (40000, )
# input_shape => (batch_size, input_dim)=(60000, 40000)인데 행 무시 하므로 (40000,) 과 같다.
model.add(Dense(32, activation='relu'))
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
                      filepath=filepath + 'CNN_1_' + date + '_' + filename)

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
# 기존 성능
<2>
Epoch 11/100
1497/1500 [============================>.] - ETA: 0s - loss: 0.0453 - acc: 0.9871Restoring model weights from the end of the best epoch: 6.
Epoch 00011: val_loss did not improve from 0.11931
1500/1500 [==============================] - 8s 6ms/step - loss: 0.0454 - acc: 0.9871 - val_loss: 0.1740 - val_acc: 0.9654
Epoch 00011: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.1190 - acc: 0.9684
loss :  0.11896849423646927
accuracy :  0.9684000015258789

# padding 적용시...
Epoch 15/100
1497/1500 [============================>.] - ETA: 0s - loss: 0.0503 - acc: 0.9849Restoring model weights from the end of the best epoch: 10.

Epoch 00015: val_loss did not improve from 0.12680
1500/1500 [==============================] - 9s 6ms/step - loss: 0.0503 - acc: 0.9849 - val_loss: 0.2156 - val_acc: 0.9610
Epoch 00015: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.1226 - acc: 0.9701
loss :  0.1225610002875328
accuracy :  0.9700999855995178
time :  144.61131191253662

: 성능이 아주 조금 좋아졌다.

# MaxPooling 적용시...
Epoch 13/100
1500/1500 [==============================] - ETA: 0s - loss: 0.0572 - acc: 0.9829Restoring model weights from the end of the best epoch: 8.

Epoch 00013: val_loss did not improve from 0.10090
1500/1500 [==============================] - 7s 5ms/step - loss: 0.0572 - acc: 0.9829 - val_loss: 0.1323 - val_acc: 0.9687
Epoch 00013: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.0914 - acc: 0.9740
loss :  0.09140654653310776
accuracy :  0.9739999771118164
time :  93.93943643569946

: 성능이 좀 더 좋아졌다.
"""

"""
<학습 내용>

■ padding 원리
convolution 과정에서 반드시 padding이 필요하다.
convolution filter를 통과하면 input 이미지가 작아지는데
이때 padding을 사용하면 그대로 유지할 수 있다.
이미지의 외곽을 빙 둘러서 1픽셀씩 더 크게 만들고
추가된 1 픽셀에 0의 값을 주며 이를 zero padding이라고 함.

즉 convolution에는 크게 2가지 종류가 있다.
1 Valid convolution : padding 없음
이미지 사이즈 변화 n x m * f x q => n-f+1 x m-q+1

2 Same convolution : 아웃풋 이미지의 사이즈 동일하게 padding
n+2p-f+1 x m+2t-q+1 = n x m 이려면
p=(f-1)/2
t=(q-1)/2

# convolution에서 padding을 하는 이유
1 아웃풋 이미지의 크기를 유지하기 위해
2 Edge 쪽 픽셀 정보를 더 잘 이용하기 위해

■ padding 하는 방법
모델 안에 Conv2D 안에 padding이 있음. Conv2D 속괄호 안에 있어야 함.
Conv2D(filter, kernel_size, input_shape, padding, activation)

■ Pooling 원리, 목적
CNN에서 pooling이란 연산 없이 특징만 뽑아내는 과정이다.
1 Max Pooling : 정해진 크기 안에서 가장 큰 값만 뽑아낸다.
2 Average Pooling : 정해진 크기 안에서 값들의 평균을 뽑아낸다.

■ MaxPooling 하는 방법
from tensorflow.keras.layers import MaxPooling2D
model.add(MaxPooling2D())

- pool_size : pooling에 사용할 filter의 크기를 정하는 것.(단순한 정수, 또는 튜플형태 (N,N))
- strides : pooling에 사용할 filter의 strides를 정하는 것.
- padding : "valide"(=padding을 안하는것) or "same"(=pooling결과 size가 input size와 동일하게 padding)

■ Stride: 성큼성큼 걷다. (칸 이동)
Conv2D 에서 stride의 기본값은 1이다.
Conv2D 에서 stride를 2로 바꾸면 MaxPooling과 비슷하지만 특징만 뽑아내는 MaxPooling과는 다르게 연산량이 늘어난다.
MaxPooling의 stride의 기본값은 2이다.
"""