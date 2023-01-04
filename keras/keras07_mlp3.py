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