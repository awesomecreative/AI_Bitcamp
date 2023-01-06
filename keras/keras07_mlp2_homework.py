import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])
y = np.array([2,4,6,8,10,12,14,16,18,20])

print(x.shape)
print(y.shape)

x = x.T

print(x.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(5, input_dim=3))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x,y,epochs=100,batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([[10,1.4,0]])
print('[10, 1.4, 0]의 결과값 : ', result)

"""
Epoch 100/100
10/10 [==============================] - 0s 443us/step - loss: 0.1649
1/1 [==============================] - 0s 92ms/step - loss: 0.1261
loss :  0.12608596682548523
[10, 1.4, 0]의 결과값 :  [[19.895498]]
"""

"""
메모
x에 세 덩어리를 넣는 것이므로 input_dim=3 으로 넣어야 함!
input_dim= x 덩어리의 개수임
"""