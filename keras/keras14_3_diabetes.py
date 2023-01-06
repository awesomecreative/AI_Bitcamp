# [과제, 실습]
# R2 0.62 이상

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np

from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
datasets = load_diabetes()
x=datasets.data
y=datasets.target

print(x)
print(x.shape) #(442, 10)
print(y)
print(y.shape) #(442, )

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=10))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(512))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'] )
model.fit(x_train, y_train, epochs=1000, batch_size=60)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print ('RMSE : ', RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

"""
Epoch 999/1000
8/8 [==============================] - 0s 1ms/step - loss: 2977.7812 - mae: 44.2632
Epoch 1000/1000
8/8 [==============================] - 0s 1ms/step - loss: 2929.5869 - mae: 43.7691
3/3 [==============================] - 0s 0s/step - loss: 2636.3167 - mae: 41.9248
RSME :  51.345075005195916
R2 :  0.5815481464842314
"""