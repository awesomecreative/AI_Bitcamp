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

"""
메모

# matplotlib.pyplot은 그림 그리는 툴을 가진 라이브러리
# scatter는 점들을 찍어준다. 여기서 (x, y)는 맨 위에 있는 원시적인 데이터를 의미한다. 즉, scatter는 점.
# model.predict(x) predict에 x값을 넣으면 x를 넣은 전체 예측값을 알려준다.
# plt.plot(x, y_predict) 라 하면 x값을 넣었을 때 훈련된 데이터의 값을 직선으로 보여준다. 즉, plot은 직선.
# plt.show()를 해야 그림 나옴.

# 왜 loss 값이 다를까?
# train set으로 fit(훈련)한 loss 값과 test set으로 evaluate한 loss 값이 다르기 때문이다.
"""