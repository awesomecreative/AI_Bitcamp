import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM
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

x_train = x_train.reshape(464809,27,2)
x_test = x_test.reshape(116203,27,2)


#2. model
model = Sequential()
model.add(LSTM(256, input_shape=(27,2), activation='relu'))
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

# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")

# filepath = './_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
#                      filepath = filepath + 'k39_10_' + date + '_' + filename)

import time
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=300, validation_split=0.2, verbose=1, callbacks=[es]) # callbacks=[es, mcp]
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
Epoch 9/1000
1240/1240 [==============================] - ETA: 0s - loss: 1.2075 - accuracy: 0.4870Restoring model weights from the end of the best epoch: 4.
1240/1240 [==============================] - 88s 71ms/step - loss: 1.2075 - accuracy: 0.4870 - val_loss: 1.2004 - val_accuracy: 0.4896
Epoch 00009: early stopping
3632/3632 [==============================] - 49s 14ms/step - loss: 0.6415 - accuracy: 0.7296
loss :  0.6415084004402161
accuracy :  0.7295852899551392
y_pred_1 :  [[8.33709896e-01 1.24096654e-01 8.96039546e-06 ... 2.88817420e-04
  1.45431511e-06 4.18941863e-02]
 [1.61337882e-01 8.33027840e-01 3.19273531e-05 ... 4.96472651e-03
  1.42519202e-04 4.94898006e-04]
 [5.86084068e-01 4.08451647e-01 2.00336162e-05 ... 1.21007918e-03
  1.87782825e-05 4.21538437e-03]
 ...
 [8.18721727e-02 9.10152495e-01 4.12467634e-05 ... 7.42884120e-03
  2.42464172e-04 2.62405374e-04]
 [5.49655942e-06 5.27074793e-03 4.22739506e-01 ... 4.31555673e-04
  1.66458502e-01 6.99864631e-06]
 [2.54871398e-02 7.12160587e-01 8.47222880e-02 ... 3.90957519e-02
  1.37366325e-01 8.33884988e-04]]
y_test_1 :  [[0. 1. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [0. 1. 0. ... 0. 0. 0.]
 ...
 [0. 1. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]]
y_pred_2 :  [0 1 0 ... 1 2 1]
y_test_2 :  [1 0 1 ... 1 3 2]
acc :  0.7295852946997926
time :  782.222638130188
"""