import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston

#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=333)

#2. model
model = Sequential()
model.add(Dense(20, input_dim=13, activation = 'linear'))
model.add(Dense(40, input_shape=(13,)))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

#3. compile, fit
model.compile(loss = 'mse', optimizer = 'adam')

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)

hist = model.fit(x_train, y_train, epochs=300, batch_size=1, validation_split=0.2, verbose=1,
                 callbacks=[earlyStopping])

#4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('RMSE: ', rmse)
print('r2: ', r2)

print("=======================================")
print(hist) # <keras.callbacks.History object at 0x00000195CE646850>
print("=======================================")
print(hist.history)
print("=======================================")
print(hist.history['loss'])
print("=======================================")
print(hist.history['val_loss'])
print("=======================================")

#5. submit, draw

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('boston loss')
plt.legend(loc='center right')
plt.show()

"""
Epoch 86/300
251/323 [======================>.......] - ETA: 0s - loss: 36.7740Restoring model weights from the end of the best epoch: 76.
323/323 [==============================] - 0s 734us/step - loss: 38.5634 - val_loss: 26.3918
Epoch 00086: early stopping
4/4 [==============================] - 0s 1ms/step - loss: 35.6511
loss :  35.65113830566406
RMSE:  5.970857473406167
r2:  0.63650577943559
"""

"""
Memo
early stopping 원리 : 기존의 최소값과 새로운 값을 비교해 최소값을 갱신한다.
from tensorflow.keras.callbacks import EarlyStopping
첫 글자가 대문자면 python의 Class에 구성되어 있다. 보통 함수는 소문자로 시작한다.
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True, verbose=1)
accuracy는 mode='max', loss 또는 val_loss는 mode='min', 잘 모르겠으면 mode='auto'로 놓는다.
patience: 갱신이 안되는 걸 몇 번 참겠다는 의미이다.
치명적인 문제점 : 원하는 시점이 아닌 끊은 시점의 weight가 저장된다.
따라서 restore_best_weights=True를 설정해줘서 끊는 시점이 아닌 가장 loss가 낮은 시점의 weight를 저장한다.
default(기본값) : resotre_best_weights=False
verbose=1 : early stopping 지점 보여줌.

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
# https://deep-deep-deep.tistory.com/55

Epoch 19/300
285/323 [=========================>....] - ETA: 0s - loss: 74.4751Restoring model weights from the end of the best epoch: 9.
323/323 [==============================] - 0s 1ms/step - loss: 73.0844 - val_loss: 51.9514
Epoch 00019: early stopping
4/4 [==============================] - 0s 3ms/step - loss: 76.8515
loss :  76.85153198242188
"""
