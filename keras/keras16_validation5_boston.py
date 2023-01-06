import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=13, activation = 'linear'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'] )
model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split=0.2)

#4. 평가, 예측
model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

"""
11/11 [==============================] - 0s 3ms/step - loss: 14.8122 - mae: 2.7673 - val_loss: 46.7388 - val_mae: 4.9632
4/4 [==============================] - 0s 694us/step - loss: 50.3519 - mae: 5.6027
RMSE :  7.095908932154797
R2 :  0.391412778161913
"""