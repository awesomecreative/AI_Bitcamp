# return_sequences=True 넣기!

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Flatten
from tensorflow.keras.layers import Conv1D

#1. data
a = np.array(range(1, 101))         # [1,2,3,4,5,6, ... ,97, 98, 99, 100]
x_predict = np.array(range(96, 106))    # [96,97,98,...,102,103,104,105]
# 예상 y = 100~106 

timesteps = 5                       # x는 4개, y는 1개

def split_x(dataset, timesteps) :
    aaa = []
    for i in range(len(dataset) - timesteps +1):
        subset = dataset[i : (i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)

a = split_x(a, timesteps)
x = a[:, :-1]
y = a[:, -1]

x_predict = split_x(x_predict, 4)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=123)


print(x_train.shape, x_test.shape)      # (72, 4) (24, 4)

x_train = x_train.reshape(72,4,1)
x_test = x_test.reshape(24,4,1)
x_predict = x_predict.reshape(7,4,1)

#2. model
model = Sequential()
# model.add(LSTM(256, input_shape=(4,1), activation='relu'))
# model.add(Bidirectional(LSTM(256, activation='relu', return_sequences=True), input_shape=(4,1)))
# model.add(LSTM(64))
model.add(Conv1D(100, 1, input_shape=(4,1)))
model.add(Conv1D(100, 2))
model.add(Conv1D(100, 2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()

#3. compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100)

#4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_predict)
print('y_predict : ', y_predict)
print(y_predict.shape)


"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv1d (Conv1D)             (None, 3, 100)            300

Epoch 100/100
3/3 [==============================] - 0s 1ms/step - loss: 770.4786
1/1 [==============================] - 0s 163ms/step - loss: 955.7688
loss :  955.768798828125
y_predict :
[[[72.20918 ] [72.55959 ]]
[[72.55959 ] [72.91012 ]]
[[72.91012 ] [73.2708  ]]
[[73.2708  ] [73.631454]]
[[73.631454] [73.992134]]
[[73.992134] [74.352776]]
[[74.35279 ] [74.71346 ]]]
(7, 2, 1)
"""

"""
<학습 내용>
■ Conv1D (1차원, 선)
시계열 자료에 많이 사용함.
Conv1D 여러번 layer 해야 잘 나옴.

■ Conv1D param#
=> 찾아서 정리하기.

■ 결과값 나오는 형태
모든 Conv1D 레이어마다 모든 커널 사이즈에서 1을 기준으로 1 추가될 때마다 기존의 값에서 -1
=> 왜 그런지는 모르겠음.
"""