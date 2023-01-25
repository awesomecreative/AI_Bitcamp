import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
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


################케라스 투카테고리컬##################
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y, y.shape, type(y)) # (581012, 8) <class 'numpy.ndarray'>
print(np.unique(y[:,0], return_counts=True)) # (array([0.], dtype=float32), array([581012], dtype=int64))
print(np.unique(y[:,-1], return_counts=True)) # (array([0., 1.], dtype=float32), array([560502,  20510], dtype=int64))
y = np.delete(y, 0, axis=1) # y의 0번째 column을 지운다.
print(y, y.shape, type(y)) # (581012, 7) <class 'numpy.ndarray'>
####################################################

# ##################판다스 겟더미스###################
# import pandas as pd
# y = pd.get_dummies(y)
# print(y, y.shape, type(y)) # (581012, 7) <class 'pandas.core.frame.DataFrame'>
# y = y.to_numpy()
# print(y, y.shape, type(y)) # (581012, 7) <class 'numpy.ndarray'>
# # y.to_numpy() 안해도 y_train은 훈련을 통과해서 pandas에서 자동으로 numpy로 바뀐다.
# # 하지만 y_test는 아직 pandas 형태이다. np.argmax 즉 numpy는 pandas 데이터를 받지 못 하기 때문에 오류가 발생한다.
# # 따라서 .to_numpy() 나 .values 를 사용해서 numpy 데이터 형태로 바꿔준다.
# ####################################################

# ##################싸이킷런 원핫인코더####################
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()
# y = y.reshape(-1,1) # reshape 할 때 중요한 점 : 데이터의 내용과 순서가 바뀌지 않아야 한다.
# print(y, y.shape, type(y)) # (581012, 1) <class 'numpy.ndarray'>
# y = ohe.fit_transform(y) # y=ohe.fit(y)와 y=ohe.transform(y)를 한 번에 적은 것임. : ohe도 훈련하는 거라 가중치 생김.
# print(y, y.shape, type(y)) # (581012, 7) <class 'scipy.sparse._csr.csr_matrix'>
# y = y.toarray()
# print(y, y.shape, type(y)) # (581012, 7) <class 'numpy.ndarray'>
# # CSR matrix 를 numpy 데이터 형태로 바꾸기 위해서는 .to_numpy(), .values 가 아니라 .toarray()이다.
# ########################################################

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
#scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (464809, 54) (116203, 54)

x_train = x_train.reshape(464809,9,2,3)
x_test = x_test.reshape(116203,9,2,3)


#2. model
model = Sequential()
model.add(Conv2D(256, (5,3), input_shape=(9,2,3), padding='same', activation='relu'))
model.add(Conv2D(128, (5,3), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(Dropout(0.5))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. compile, fit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True, verbose=1)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                     filepath = filepath + 'k39_10_' + date + '_' + filename)

import time
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.2, verbose=1, callbacks=[es, mcp])
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

y_predict = np.argmax(y_predict, axis=1)
print('y_pred_2 : ', y_predict)
print(type(y_predict))

y_test = np.argmax(y_test, axis=1)
print('y_test_2 : ', y_test)
print(type(y_test))

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)
print('time : ', end-start)

"""
Epoch 33/1000
3716/3719 [============================>.] - ETA: 0s - loss: 0.1872 - accuracy: 0.9249Restoring model weights from the end of the best epoch: 28.

Epoch 00033: val_loss did not improve from 0.19016
3719/3719 [==============================] - 28s 8ms/step - loss: 0.1872 - accuracy: 0.9249 - val_loss: 0.1970 - val_accuracy: 0.9222
Epoch 00033: early stopping
3632/3632 [==============================] - 12s 3ms/step - loss: 0.1903 - accuracy: 0.9235
loss :  0.19033066928386688
accuracy :  0.9235131740570068
y_pred_1 :  [[1.6026275e-01 8.3720684e-01 7.5356118e-05 ... 1.3204583e-05
  1.0740006e-04 2.3339048e-03]
 [7.2523355e-01 2.7453184e-01 2.1401549e-06 ... 7.2235642e-05
  4.9698297e-06 1.5514703e-04]
 [4.1531608e-02 9.5763361e-01 1.2367308e-05 ... 7.4782281e-04
  4.1720763e-05 3.2483116e-05]
 ...
 [2.2467445e-03 9.8624551e-01 3.4473513e-04 ... 1.0945681e-02
  1.8382349e-04 3.1594045e-05]
 [7.9385791e-05 1.3564828e-05 6.6548097e-03 ... 2.0277874e-05
  2.5900081e-02 4.1483422e-06]
 [1.2268539e-01 2.8339979e-01 4.4354978e-01 ... 8.5841820e-02
  6.3598685e-02 2.6317075e-04]]
y_test_1 :  [[0. 1. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [0. 1. 0. ... 0. 0. 0.]
 ...
 [0. 1. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]]
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
y_pred_2 :  [1 0 1 ... 1 3 2]
<class 'numpy.ndarray'>
y_test_2 :  [1 0 1 ... 1 3 2]
<class 'numpy.ndarray'>
acc :  0.9235131623107837
time :  945.9347672462463
"""