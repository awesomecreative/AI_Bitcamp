# 싸이킷런 원핫엔코더
# from sklearn.preprocessing import OneHotEncoder


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

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
print(y) # [5 5 2 ... 3 3 3]
print(y.shape) # (581012,)
print(type(y)) # <class 'numpy.ndarray'>

y = y.reshape(-1, 1)

print(y) # [[5] [5] [2] ... [3] [3] [3]]
print(y.shape) # (581012, 1)
print(type(y)) # <class 'numpy.ndarray'>

y = ohe.fit_transform(y)
ohe = OneHotEncoder(sparse=False)
print(y)
#   (0, 4)        1.0
#   (1, 4)        1.0
#   (2, 1)        1.0
#   (3, 1)        1.0
#   :     :
#   (581010, 2)   1.0
#   (581011, 2)   1.0
print(y.shape) # (581012, 7)
print(type(y)) # <class 'scipy.sparse._csr.csr_matrix'>

y = y.toarray()
print(y)
# [[0. 0. 0. ... 1. 0. 0.]
#  [0. 0. 0. ... 1. 0. 0.]
#  [0. 1. 0. ... 0. 0. 0.]
#  ...
#  [0. 0. 1. ... 0. 0. 0.]
#  [0. 0. 1. ... 0. 0. 0.]
#  [0. 0. 1. ... 0. 0. 0.]]
print(y.shape) # (581012, 7)
print(type(y)) # <class 'numpy.ndarray'>

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

y_predict = np.argmax(y_predict, axis=1)
print('y_pred_2 : ', y_predict)
y_test = np.argmax(y_test, axis=1)
print('y_test_2 : ', y_test)
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)
print('time : ', end-start)

"""
Epoch 18/100
3709/3719 [============================>.] - ETA: 0s - loss: 0.6337 - accuracy: 0.7239Restoring model weights from the end of the best epoch: 13.
3719/3719 [==============================] - 4s 1ms/step - loss: 0.6337 - accuracy: 0.7238 - val_loss: 0.6415 - val_accuracy: 0.7244
Epoch 00018: early stopping
3632/3632 [==============================] - 2s 550us/step - loss: 0.6116 - accuracy: 0.7379
loss :  0.6116142868995667
accuracy :  0.7378811240196228
y_pred_1 :  [[3.06416042e-02 9.58142579e-01 5.05572744e-03 ... 2.56969384e-03
  3.44717572e-03 1.42910474e-04]
 [7.09988534e-01 2.87606806e-01 5.79313735e-07 ... 4.83423064e-05
  6.79198138e-06 2.34901183e-03]
 [1.04106439e-04 3.86198722e-02 6.97105706e-01 ... 1.48951833e-03
  2.29384169e-01 1.19130791e-05]
 ...
 [8.00227284e-01 1.61981910e-01 1.03836406e-07 ... 3.30756575e-06
  5.14671592e-06 3.77822556e-02]
 [7.48449028e-01 2.50592530e-01 5.29267825e-04 ... 3.50794144e-04
  6.17798214e-05 1.65407619e-05]
 [8.03334057e-01 1.96018040e-01 1.33966962e-08 ... 2.29114848e-05
  7.80114135e-07 6.24152948e-04]]
y_test_1 :  [[0. 1. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 1. 0.]
 ...
 [1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]]
y_pred_2 :  [1 0 2 ... 0 0 0]
y_test_2 :  [1 0 5 ... 0 0 0]
acc :  0.7378811218299012
time :  83.41472578048706
"""

"""
<scikit-onehotencoder vs pandas-get_dummies vs keras-to_categorical>

OneHotEncoder : 명목변수든 순위변수든 모두 원핫인코딩을 해준다.
=> 해결방법: shape 맞추기

0) scikit-learn에서 OneHotEncoder 가져오기
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)

1) 스칼라: 원본 데이터를 y, y.shape, type(y)를 print 해보면
(581012,) 스칼라 형태의 numpy.ndarray 임을 알 수 있다.

2) 벡터: 원핫엔코더하려면 벡터 형태로 reshape 해줘야 한다.
y = y.reshape(-1,1) 해서 (581012, 1) 벡터 형태의 numpy.ndarray를 만든다.
# (-1,1) 하면 (전체, 1)과 같다.

3) 원핫엔코딩: y = ohe.fit_transform(y)로 원핫엔코딩한다.
y = ohe.fit_transform(y) 하면 (581012, 7) 벡터 형태의 scipy.sparse._csr.csr_matrix가 나온다.

4) 데이터형태 바꾸기 : scipy CSR matrix 를 Numpy ndarray로 바꾼다.
y = y.toarray() 하면 데이터 종류만 numpy ndarray로 바뀐다.
"""