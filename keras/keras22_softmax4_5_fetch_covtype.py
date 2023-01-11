# <원핫인코딩 최종 정리>

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


# ################케라스 투카테고리컬##################
# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y, y.shape, type(y)) # (581012, 8) <class 'numpy.ndarray'>
# print(np.unique(y[:,0], return_counts=True)) # (array([0.], dtype=float32), array([581012], dtype=int64))
# print(np.unique(y[:,-1], return_counts=True)) # (array([0., 1.], dtype=float32), array([560502,  20510], dtype=int64))
# y = np.delete(y, 0, axis=1) # y의 0번째 column을 지운다.
# print(y, y.shape, type(y)) # (581012, 7) <class 'numpy.ndarray'>
# ####################################################

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
model.fit(x_train, y_train, epochs=1, batch_size=100, validation_split=0.2, verbose=1, callbacks=[earlyStopping])
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