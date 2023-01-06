import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10),range(21,31),range(201,211)])
# print(range(10))
print(x.shape)      # (3,10)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]])

x=x.T
y=y.T
print(x.shape)
print(y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(2))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=3)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss : ', loss)

result = model.predict([[9,30,210]])
print('[9,30,210]의 값은 : ', result)

"""
Epoch 500/500
4/4 [==============================] - 0s 673us/step - loss: 1.0194
1/1 [==============================] - 0s 73ms/step - loss: 0.2699
loss :  0.2698874771595001
[9,30,210]의 값은 :  [[9.882986  1.9956133]]
"""

"""
메모

# 결과값이 2개가 나와야 하므로 마지막 나오는 값 레이어를 2라 해야한다.

# Q) 모델 추가할 때 input은 dimension을 넣는데 왜 output은 dimension을 지정 안 해주는 지 궁금합니다.
# A) 마지막 output은 마지막 Dense에서 지정해주기 때문에 굳이 output_dim을 지정할 필요가 없기 때문이다.

# Q) 마지막 predict 값에 (1,3) 행렬을 넣는 이유는?
# A) (10,3) : 행 무시 열 우선 : 즉, 행의 개수는 달라도 되지만 넣은 값과 나오는 값의 열의 개수를 동일하게 맞춰야한다.
# 여기에서 (10,3)의 행렬을 넣었으므로 예측값에는 (아무 값, 3)의 행렬을 넣어야한다.

# range(10)은 [0,1,2,3,4,5,6,7,8,9]를 의미한다.
# range(21,31)은 [21,22,23,24,25,26,27,28,29,30]을 의미한다.

# x=x.T, y=y.T 하는 이유는 행과 열을 바꿔서 (3,10)과 (2,10)을 (10,3) (10,2)로 해줘야 하기 때문이다.
# 행렬 표시할 때 (1,3)은 1행 3열을 의미한다. (행, 열)임.
"""