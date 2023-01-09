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
model.add(Dense(5, input_dim=13, activation = 'linear'))
model.add(Dense(5, input_shape=(13,)))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. compile, fit
import time
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae', 'mse'])
start = time.time()
model.fit(x_train, y_train, epochs=50, batch_size=1, validation_split=0.2, verbose=2)
end = time.time()

#4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print("time : ", end-start)

#5. submit



"""
Memo
<input_dim, input_shape>
input_dim 은 오직 (행,렬)에서만 사용가능하다.
input_shape 는 다차원 ( , , , , ...)에서도 사용 가능하다.
예를 들어, (100, 10, 5)의 경우 (10,5)가 100개 있다는 뜻이고
행무시 열우선이므로 맨 앞에 있는 100을 제외하고 input_shape에 (10,5)를 넣는다.

<verbose: 말 수가 많은>
verbose=0 : 과정을 출력 안 한다.
verbose=1 : 과정을 자세히 출력함. progress bar가 있다. [=========>]
verbose=2 : 과정을 함축적으로 출력함. progress bar가 없다. 대신 epoch당, step당 걸린 시간을 표현해준다.
verbose=3 이상 : 과정에서 epoch 숫자만 보여준다.
걸리는 시간은 verbose=1이 나머지에 비해 크다.
default(기본값)은 verbose=1 이다.

# 스칼라 0차원, 벡터 1차원, 매트릭스 2차원, 텐서 3차원 이상
"""