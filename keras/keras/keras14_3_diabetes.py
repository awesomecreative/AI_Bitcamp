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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=123)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=10))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(60))
model.add(Dense(100))
model.add(Dense(60))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'] )
model.fit(x_train, y_train, epochs=100, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RSME(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print ('RSME : ', RSME(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

"""
Epoch 99/100
11/11 [==============================] - 0s 999us/step - loss: 2937.5986 - mae: 43.8780
Epoch 100/100
11/11 [==============================] - 0s 970us/step - loss: 2895.4133 - mae: 43.4398
4/4 [==============================] - 0s 258us/step - loss: 2921.9883 - mae: 43.8207
RSME :  54.05541978170263
R2 :  0.5085354858326381
"""