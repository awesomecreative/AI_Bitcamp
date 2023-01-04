from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,21)) # (20, )
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20]) # (20, )


x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123                                                    
)

print('x_train : ', x_train)
print('x_test : ', x_test)
print('y_train : ', y_train)
print('y_test : ', y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x)

import matplotlib.pyplot as plt
plt.scatter(x, y) 
plt.plot(x, y_predict, color='red')
plt.show()

"""
Epoch 99/100
14/14 [==============================] - 0s 458us/step - loss: 1.8871
Epoch 100/100
14/14 [==============================] - 0s 533us/step - loss: 1.8972
1/1 [==============================] - 0s 78ms/step - loss: 3.1369
loss :  3.136857271194458
"""
