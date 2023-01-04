
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

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=123)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim=8))
model.add(Dense(512))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train,y_train, epochs=100, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

"""
Epoch 99/100
452/452 [==============================] - 0s 876us/step - loss: 0.6568 - mae: 0.6073
Epoch 100/100
452/452 [==============================] - 0s 843us/step - loss: 0.6641 - mae: 0.6095
194/194 [==============================] - 0s 633us/step - loss: 0.7025 - mae: 0.6772
RMSE :  0.8381682702778236
R2 :  0.46870464625147235
"""