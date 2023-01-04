import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

# RMSE는 낮을수록, R2가 높아지면 성능이 좋다고 판단할 수 있다.
# 보통 성능 지표는 2가지 이상을 본다.

"""
Epoch 99/100
14/14 [==============================] - 0s 539us/step - loss: 10.2418 - mae: 2.1288
Epoch 100/100
14/14 [==============================] - 0s 567us/step - loss: 11.0021 - mae: 2.4521
1/1 [==============================] - 0s 94ms/step - loss: 14.8518 - mae: 2.9355
loss :  [14.851837158203125, 2.9354991912841797]
================================
[ 9  7  5 23  8  3]
[[13.569388 ]
 [ 5.7303953]
 [ 4.8593965]
 [16.182388 ]
 [ 8.343394 ]
 [ 7.472393 ]]
================================
RMSE :  3.853808159368706
R2 :  0.6475503336507545
"""
