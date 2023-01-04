import tensorflow as tf
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1,input_dim=1))

#3. 컴파일,훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=2000)

#4. 평가,예측
result = model.predict([4])
print('결과 : ', result)

"""
Epoch 2000/2000
1/1 [==============================] - 0s 997us/step - loss: 1.5712e-04
결과 :  [[4.000495]]
"""