# 케라스 투카테고리컬
# from tensorflow.keras.utils import to_categorical

import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. data
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)
print(np.unique(y, return_counts=True))
#(581012, 54) (581012,)
#(array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
# => shape와 np.unique를 확인한 결과 이 데이터는 회귀가 아닌 다중분류이다.

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(np.unique(y[:,0], return_counts=True)) # y값 전체 중 0번째 컬럼의 고유한 값 리턴
print(np.unique(y[:,1], return_counts=True)) # y값 전체 중 1번째 컬럼의 고유한 값 리턴
print(np.unique(y[:,-1], return_counts=True)) # y값 전체 중 마지막 컬럼의 고유한 값 리턴

y = np.delete(y, 0, axis=1) # y의 0번째 컬럼을 지운다.
print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1)

#2. model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(54,)))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(120, activation='linear'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. compile, fit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True, verbose=1)
import time
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[earlyStopping])
end = time.time()

#4. evaluate, predict
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
print('y_pred_1 : ', y_predict)
print('y_test_1 : ', y_test)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
print('y_pred_2 : ', y_predict)
print('y_test_2 : ', y_test)
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)
print('time : ', end-start)

"""
Epoch 13/100
11620/11621 [============================>.] - ETA: 0s - loss: 0.7425 - accuracy: 0.6800Restoring model weights from the end of the best epoch: 8.
11621/11621 [==============================] - 12s 1ms/step - loss: 0.7425 - accuracy: 0.6800 - val_loss: 0.7166 - val_accuracy: 0.6942
Epoch 00013: early stopping
3632/3632 [==============================] - 2s 558us/step - loss: 0.6966 - accuracy: 0.7040
loss :  0.6966418623924255
accuracy :  0.7040265798568726
y_pred_1 :  [[6.63413703e-02 8.72235298e-01 2.91118696e-02 ... 1.63661130e-02
  1.57207828e-02 2.18994232e-04]
 [5.37476540e-01 4.41292405e-01 1.15550924e-04 ... 1.08999223e-03
  1.02201775e-05 2.00152714e-02]
 [2.53470716e-06 1.85067467e-02 6.37353361e-01 ... 1.23882288e-04
  2.02319190e-01 4.08861088e-05]
 ...
 [6.49081051e-01 3.36810768e-01 2.98866526e-05 ... 1.33304379e-03
  8.78484116e-06 1.27365142e-02]
 [1.54853493e-01 8.14902246e-01 7.42991595e-03 ... 1.32439509e-02
  8.94334167e-03 6.25513960e-04]
 [8.12714934e-01 1.84170395e-01 7.39659981e-07 ... 9.31245682e-04
  2.47538856e-06 2.18006293e-03]]
y_test_1 :  [[0. 1. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 1. 0.]
 ...
 [1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]]
y_pred_2 : [1 0 2 ... 0 1 0]
y_test_2 : [1 0 5 ... 0 0 0]
acc :  0.7040265741848317
time :  155.02043437957764
"""

"""
Memo
<scikit-onehotencoder vs pandas-get_dummies vs keras-to_categorical>
to_categorical의 특성 : 무조건 0부터 시작하게끔 한다. => 0이 없을 경우 class 하나 더 만듦.
y 데이터가 [1 2 3 4 5 6 7]일 경우 to_categorical(y)하면 [0 1 2 3 4 5 6 7]로 0을 더 추가해 만듦.

=> 확인하는 방법: np.unique, y[:,0], y[:,-1], return_counts=True
print(np.unique(y[:,0], return_counts=True)) # y값 전체 중 0번째 컬럼의 고유한 값 리턴
print(np.unique(y[:,1], return_counts=True)) # y값 전체 중 1번째 컬럼의 고유한 값 리턴
print(np.unique(y[:,-1], return_counts=True)) # y값 전체 중 마지막 컬럼의 고유한 값 리턴

=> 해결방법: 첫번째 칼럼 삭제하기!
=> y = np.delete(y, 0, axis=1)
np.delete(데이터, 0번째, 행삭제는 axis=0, 열삭제는 axis=1)

<datasets 다운이 잘 안 받아질 때>
from sklearn import datasets
print(datasets.get_data_home())
from sklearn.datasets import fetch_covtype, load_wine
"""