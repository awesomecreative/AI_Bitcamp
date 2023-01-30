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

"""
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

"""
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
■ Conv1D vs Conv2D
Conv1D : time series, sequence와 같은 1차원 데이터에 사용함.
Conv2D : image와 같은 2차원 데이터에 사용함.
=> 둘 다 여러 번 layer 해야 잘 나온다.

■ Conv1D의 input_shape
3차원 (m,n,p) 데이터의 Conv1D layer의 input_shape는 행무시 열우선으로 (n,p)이다.
원래 첫번째 Conv1D layer의 input_shape = (m,1)로 열이 1인 것이 정석이다.
그런데 input_shape를 (m,1) 대신 (m,n) 으로 넣어도 모델이 잘 작동된다. 오히려 성능이 좋을 때도 있다.
왜 그럴까? 입력 데이터에 여러 채널 또는 기능이 있을 경우에 input_shape=(m,n)을 사용하면
첫 번째 Conv1D 레이어에서 각 채널 또는 기능을 독립적으로 처리할 수 있어
데이터에서 더 많은 정보와 복잡한 패턴을 캡처하는 데 도움이 될 수 있다.
하지만 input_shape = (m,n)을 사용하면 모델의 매개 변수 수와 계산의 복잡성이 증가하기 때문에
데이터와 문제에 가장 적합한 입력 모양을 선택하는 것이 중요하다.

■ Conv1D param#
Conv1D param# = (kernel_size * input_channels + 1) * number_of_filters
Conv1D(filters, kernel_size, input_shape=( , ))
input_shape = (m,n) = (input_length, input_channels)
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv1d (Conv1D)             (None, 4, 100)            200

 conv1d_1 (Conv1D)           (None, 3, 100)            20100

 conv1d_2 (Conv1D)           (None, 2, 100)            20100

 flatten (Flatten)           (None, 200)               0

예시) Conv1D(100, 1, input_shape=(4,1))
Conv1D param# = (1 * 1 + 1) * 100 = 200

■ Flatten param#
Flatten param# = 0
단순히 다차원을 2차원으로 바꿔주는 역할이라 연산량은 없다.
=> Dense에 넣기 좋은 형태로 바꿔줌.

■ 결과값 나오는 형태
모든 Conv1D 레이어마다 모든 커널 사이즈에서 1을 기준으로 1 추가될 때마다 기존의 값에서 -1
=> 이유) Conv2D랑 똑같이 kernel_size를 단위로 잘라서 줄어드는 것임.
=> input_shape = (m, 1) 을 filters = e, kernel_size = q 에 통과하면, output_shape = (None, m-q+1 ,e) 이다.
"""