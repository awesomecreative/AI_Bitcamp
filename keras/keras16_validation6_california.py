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

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=123)

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=8, activation = 'linear'))
model.add(Dense(64, activation = 'sigmoid'))
model.add(Dense(128, activation = 'sigmoid'))
model.add(Dense(512, activation = 'sigmoid'))
model.add(Dense(64, activation = 'sigmoid'))
model.add(Dense(16, activation = 'sigmoid'))
model.add(Dense(1, activation = 'linear'))

#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
start = time.time()
model.fit(x_train,y_train, epochs=500, batch_size=50, validation_split=0.2)
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
265/265 [==============================] - 0s 1ms/step - loss: 0.4411 - mae: 0.4866 - val_loss: 0.5124 - val_mae: 0.5065
Epoch 500/500
265/265 [==============================] - 0s 1ms/step - loss: 0.4280 - mae: 0.4780 - val_loss: 0.4656 - val_mae: 0.5066
129/129 [==============================] - 0s 1ms/step - loss: 0.4785 - mae: 0.5146
RMSE :  0.6917556103687497
R2 :  0.6401558631946828
time :  193.5263843536377
"""