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

"""
메모

# (10,)이거나 (10,1)은 input_dim=1로 입력 가능하다.
# ex) 비트코인의 가격을 가지고 환율, 금리, 신생아 출생수를 구하고자 할 때
# 문제 : x, y로 훈련했는데 평가 때에도 x, y로 평가함. 즉, 같은 값으로 평가함.
# 해결 방법: 평가할 때에는 훈련했을 때에와 다른 값으로 평가해야함.
# 해결 방법: 데이터를 쪼개서 70~80%의 데이터로 훈련시키고 나머지 20%의 데이터로 평가시킨다.
# 데이터를 받으면 train set, validation set, test set으로 나누기.
"""