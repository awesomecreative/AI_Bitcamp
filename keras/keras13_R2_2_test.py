# 실습
#1. R2를 음수가 아닌 0.5 이하로 줄이기
#2. 데이터는 건들지 말 것
#3. 레이어는 임풋 아웃풋 포함 7개 이상
#4. batch_size=1
#5. 히든레이어의 노드는 각각 10개 이상 100개 이하
#6. train 70%
#7. epoch 100번 이상
#8. loss지표는 mse 또는 mae
# [실습시작]
# 즉, R2를 강제적으로 나쁘게 만들어라.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,21)) # (20, )
y = np.array(range(1,21)) # (20, )

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123                                                    
)

print('x_train : ', x_train)
print('x_test : ', x_test)
print('y_train : ', y_train)
print('y_test : ', y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

print("================================")
print(y_test)
print(y_predict)
print("================================")

from sklearn.metrics import mean_squared_error, r2_score
def RMSE(y_test, y_predict):
        return np.sqrt(mean_squared_error(y_test, y_predict))
    
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

"""
Epoch 100/100
14/14 [==============================] - 0s 539us/step - loss: 2.8310e-12 - mae: 1.3368e-06
1/1 [==============================] - 0s 86ms/step - loss: 5.1159e-12 - mae: 1.6689e-06
loss :  [5.115907697472721e-12, 1.6689300537109375e-06]
================================
[15  6  5 18  9  8]
[[14.999995 ]
 [ 6.       ]
 [ 4.9999995]
 [17.999994 ]
 [ 8.999999 ]
 [ 7.9999995]]
================================
RMSE :  3.0779706209559573e-06
R2 :  0.9999999999995784
"""

# e-06은 10의 -6승이라는 뜻이다.
# 