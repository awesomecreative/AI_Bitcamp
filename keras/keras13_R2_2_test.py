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
14/14 [==============================] - 0s 2ms/step - loss: 42.0035 - mae: 5.6327
1/1 [==============================] - 0s 238ms/step - loss: 22.4746 - mae: 4.1974
loss :  [22.474586486816406, 4.197406768798828]
================================
[15  6  5 18  9  8]
[[10.093409]
 [10.092798]
 [10.09273 ]
 [10.093613]
 [10.093   ]
 [10.092936]]
================================
RMSE :  4.740736948994184
R2 :  -0.0001052231200400211
"""

"""
Epoch 100/100
14/14 [==============================] - 0s 2ms/step - loss: 36.1544 - mae: 5.3187
1/1 [==============================] - 0s 229ms/step - loss: 20.5933 - mae: 3.6277
loss :  [20.59330177307129, 3.627704620361328]
================================
[15  6  5 18  9  8]
[[9.513895 ]
 [8.932351 ]
 [8.867735 ]
 [9.707744 ]
 [9.126197 ]
 [9.0615835]]
================================
RMSE :  4.537984220987326
R2 :  0.08361084247530515
"""

# e-06은 10의 -6승이라는 뜻이다.