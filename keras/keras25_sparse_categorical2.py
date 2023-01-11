# <sparse_categorical_crossentropy> : fetch_covtype => 실행 안 됨

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
print(x, x.shape, type(x))
print(y, y.shape, type(y))
print(np.unique(y, return_counts=True))
# (581012, 54) (581012,)
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
# => shape와 np.unique를 확인한 결과 이 데이터는 회귀가 아닌 다중분류이다.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1, stratify=y)

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
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import time
start = time.time()
model.fit(x_train, y_train, epochs=1, batch_size=100, validation_split=0.2, verbose=1)
end = time.time()

#4. evaluate, predict
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print('argmax 후의 y_pred : ', y_predict)

# y_test = np.argmax(y_test, axis=1)

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)
print('time : ', end-start)
