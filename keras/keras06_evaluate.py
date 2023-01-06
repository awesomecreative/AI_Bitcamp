import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x=np.array([1,2,3,4,5,6])
y=np.array([1,2,3,5,4,6])

#2. 모델구성
model=Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=2)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
result = model.predict([7])
print('7의 결과 : ', result)

"""
Epoch 200/200
3/3 [==============================] - 0s 996us/step - loss: 0.5381
1/1 [==============================] - 0s 84ms/step - loss: 0.5377
loss :  0.5376748442649841
7의 결과 :  [[6.0214033]]
"""

"""
메모
evaluate 함수는 오차값(loss)와 정확도(accuracy)를 알려준다.
항상 판단의 기준은 predict로 나온 결과값 비교가 아니라 evaluate로 나온 loss(오차)로 판단한다.
loss는 오차값으로 0과 1사이 값이며 0에 가까울수록 오차가 작은 것이다.
"""