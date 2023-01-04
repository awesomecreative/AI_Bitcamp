import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array(range(10))
# print(range(10))
print(x.shape)      # (3,10)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
             [9,8,7,6,5,4,3,2,1,0]])

x=x.T
y=y.T
print(x.shape)
print(y.shape)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(3))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=500, batch_size=5)

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([[9]])
print('[9]의 값은 : ', result)

"""
Epoch 499/500
2/2 [==============================] - 0s 998us/step - loss: 0.0859
Epoch 500/500
2/2 [==============================] - 0s 997us/step - loss: 0.0694
1/1 [==============================] - 0s 69ms/step - loss: 0.0646
loss :  0.06455440819263458
[9]의 값은 :  [[10.05252     1.6500077  -0.02796686]]
"""