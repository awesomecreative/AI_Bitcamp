# 판다스 겟더미
# from pandas as pd / y = pd.get_dummies(y)

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
# (581012, 54) (581012,)
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
# => shape와 np.unique를 확인한 결과 이 데이터는 회귀가 아닌 다중분류이다.

import pandas as pd
y = pd.get_dummies(y)
print(y)
print(y.shape) # (581012, 7)

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
model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, verbose=1, callbacks=[earlyStopping])
end = time.time()

#4. evaluate, predict
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
print('y_pred_1 : ', y_predict)
print('y_test_1 : ', y_test)
print(type(y_predict))
print(type(y_test))

import pandas as pd
y_test = y_test.values
# y_test = y_test.to_numpy() 로 해줘도 된다.
print(y_test)
print(type(y_test))

y_predict = np.argmax(y_predict, axis=1)
print('y_pred_2 : ', y_predict)
print(type(y_predict))

y_test = np.argmax(y_test, axis=1)
print('y_test_2 : ', y_test)

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)
print('time : ', end-start)

"""
Epoch 19/100
3715/3719 [============================>.] - ETA: 0s - loss: 0.6448 - accuracy: 0.7237Restoring model weights from the end of the best epoch: 14.
3719/3719 [==============================] - 21s 6ms/step - loss: 0.6448 - accuracy: 0.7237 - val_loss: 0.6661 - val_accuracy: 0.7110
Epoch 00019: early stopping
3632/3632 [==============================] - 12s 3ms/step - loss: 0.6347 - accuracy: 0.7302
loss :  0.6347299814224243
accuracy :  0.7301962971687317
y_pred_1 :  [[5.6877550e-02 9.3351710e-01 1.1296461e-03 ... 7.9496643e-03
  5.1714160e-04 8.9582973e-06]
 [8.1709611e-01 1.8264398e-01 2.1822705e-08 ... 9.6634722e-05
  1.4031903e-07 1.6312469e-04]
 [4.9772342e-07 2.4501460e-03 5.7478946e-01 ... 1.2465301e-06
  2.4387735e-01 3.7326914e-04]
 ...
 [8.7189668e-01 1.1934310e-01 2.4147232e-06 ... 1.2747593e-04
  4.5064826e-07 8.6299209e-03]
 [7.7462018e-01 2.2418836e-01 2.0796738e-06 ... 4.3037708e-04
  3.4175785e-06 7.5561419e-04]
 [6.3562179e-01 3.6370158e-01 2.5449101e-06 ... 6.0301507e-04
  3.0047083e-06 6.8025256e-05]]
y_test_1 :          1  2  3  4  5  6  7
376969  0  1  0  0  0  0  0
59897   1  0  0  0  0  0  0
247100  0  0  0  0  0  1  0
111532  0  1  0  0  0  0  0
522294  1  0  0  0  0  0  0
...    .. .. .. .. .. .. ..
72376   0  1  0  0  0  0  0
93646   0  1  0  0  0  0  0
180759  1  0  0  0  0  0  0
561349  1  0  0  0  0  0  0
209740  1  0  0  0  0  0  0

[116203 rows x 7 columns]
y_pred_2 :  [1 0 2 ... 0 0 0]
y_test_2 :  [1 0 5 ... 0 0 0]
acc :  0.7301962944158068
time :  375.25569462776184
"""

"""
<scikit-onehotencoder vs pandas-get_dummies vs keras-to_categorical>
get_dummies : 명목변수만 원핫인코딩을 해준다.

=> 해결방법: 자료형 확인
=> print(type()) 으로 자료형을 확인
y_predict는 <class 'numpy.ndarray'>
y_test는 <class 'pandas.core.frame.DataFrame'>가 나온다.

즉, y_test의 Dataframe을 numpy.ndarray로 바꿔줘야한다.
=> .values 로 pandas DataFrame을 Numpy ndarray로 바꿔주거나
=> .to_numpy() 로 pandas DataFrame을 Numpy ndarray로 바꿔주기.
"""