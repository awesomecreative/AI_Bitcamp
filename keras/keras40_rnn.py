import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. data
dataset = np.array([1,2,3,4,5,6,7,8,9,10]) # 주식 자료 11일에는? # y = ??

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])

y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(7,3,1)
print(x)
# [[[1],[2],[3]],[[2],[3],[4]],[[3],[4],[5]],[[4],[5],[6]],[[5],[6],[7]],[[6],[7],[8]],[[7],[8],[9]]]

print(x.shape) # (7,3,1)

#2. model
model = Sequential()
model.add(SimpleRNN(64, input_shape=(3,1), activation='relu')) # 행무시 열우선
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))


#3. compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

#4. evaluate, predict
loss = model.evaluate(x, y)
print('loss : ', loss)

y_pred = np.array([8,9,10]).reshape(1,3,1)
result = model.predict(y_pred)
print('[8,9,10]의 결과 : ', result)

"""
Epoch 1000/1000
1/1 [==============================] - 0s 9ms/step - loss: 2.1901e-07
1/1 [==============================] - 0s 142ms/step - loss: 2.1231e-07
loss :  2.1230577829101094e-07
[8,9,10]의 결과 :  [[11.032604]]
"""

"""
<학습 내용>
predict에서의 input_shape도 model.add에 들어가는 input_shape과 동일해야하며 행무시 열우선이므로 앞에 1 추가해야한다.

■ RNN (Recurrent Neural Network)
순환 신경망은 시계열 또는 자연어와 같은 순차적 데이터를 처리할 수 있는 신경망의 한 유형이다.
RNN은 이전 입력에 대한 정보를 유지할 수 있는 메모리가 있다.
즉, RNN은 상태를 계산할 때 이전 상태를 사용하는 피드백 루프를 네트워크에 도입함으로써 작동한다.

기본 구성 요소는 순환 뉴런이다. 반복 뉴런의 각 복사본은 자체 가중치와 편향 세트 가지며 훈련 과정 중에 업데이트된다.
역전파 알고리듬은 가중치와 편향에 대한 손실 함수의 gradients를 계산하는 데 사용되며,
이러한 gradients는 가중치와 편향을 업데이트하는 데 사용된다.


DNN은 1 2 3 4 5 6 X 에서 1~6 까지 계산해 y=x 함수 그려 X 예측함
RNN은 1 2 3 4 5 6 X 에서

  X     Y
1 2 3 | 4	=> 1에서2, 2에서3, 3에서4 찾음
2 3 4 | 5
3 4 5 | 6
4 5 6 | X

Y1    Y2    Y3
H1 -> H2 -> H3
W1    W2    W3     
X1 -> X2 -> X3

H1 = X1W1 + B
H2 = X2W2 + X1W1 + B
H3 = X3W3 + X2W2 + X1W1 + B

이런 식으로 계산함

다음 뉴런으로 넘겨줄 때 tan 함수로 처리하기 때문에 값이 많이 커지진 않는다.

# DNN 2차원 이상 input_shape
# CNN 4차원 이상 input_shape 행무시 열우선
# RNN 3차원 이상 input_shape 행무시 열우선 predict 앞에 추가, x.reshape 뒤에 추가
"""