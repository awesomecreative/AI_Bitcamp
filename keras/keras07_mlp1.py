import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])
y = np.array([2,4,6,8,10,12,14,16,18,20])

print(x.shape) # (2,10)
print(y.shape) # (10,)

x = x.T
print(x.shape) # (10,2)

#2. 모델구성
model=Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([[10,1.4]])
print('[10, 1.4]의 예측값 : ', result)

"""
Epoch 100/100
10/10 [==============================] - 0s 554us/step - loss: 0.0352
1/1 [==============================] - 0s 75ms/step - loss: 0.0421
loss :  0.0420885905623436
[10, 1.4]의 예측값 :  [[19.93524]]
"""

"""
메모

mlp는 multiple을 의미한다. 다중 입력값을 의미한다.
입력값에 두 덩어리를 넣으려면 [] bracket을 바깥에 해줘야한다.
예를 들어, [[1,2],[3,4]] 이렇게 말이다.
이렇게 두 덩어리를 넣는 예로는 입력이 환율과 금리(이자)이고 출력이 비트코인의 가격이라 생각하면 된다.

데이터.shape 하면 그 행렬의 구조가 나온다.
행렬=(행row, 열column), 행거는 가로, 열쇠는 세로
데이터.T는 행과 열을 바꿔준다. (T: 전이)

예측값 넣을때에도 [[10, 1.4]] 이렇게 괄호 2개 넣어주는 것 주의하기!
행 무시, 열 우선
열 = fiture, column, 특성
evaluate의 batch_size의 default 기본값도 32이다.
그래서 마지막에 1/1 나온다.
"""