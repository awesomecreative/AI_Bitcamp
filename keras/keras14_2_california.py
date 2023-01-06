# [실습]
# R2 0.55~0.6 이상

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

print(x)
print(x.shape) # (20640, 8)
print(y)
print(y.shape) # (20640, )

print(dataset.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(dataset.DESCR)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.75, shuffle=True, random_state=44)

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=8))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(512))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
start = time.time()
model.fit(x_train,y_train, epochs=500, batch_size=50)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

print('time : ', end-start)

"""
Epoch 499/500
310/310 [==============================] - 0s 1ms/step - loss: 0.6334 - mae: 0.5897
Epoch 500/500
310/310 [==============================] - 0s 1ms/step - loss: 0.6269 - mae: 0.5864
162/162 [==============================] - 0s 815us/step - loss: 0.6046 - mae: 0.5732
RMSE :  0.7775492374228068
R2 :  0.5514113715075821
"""

"""
메모
# 모델 성능 좋게 바꾸는 법
# 1 train_size와 random state 바꾼다.
# 2 모델 구성시 적절한 layer 층과 적절한 노드를 사용한다.
# 3 훈련 횟수와 batch_size를 조절한다.
"""
