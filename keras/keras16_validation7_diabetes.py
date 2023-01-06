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
model.add(Dense(16, input_dim=10, activation = 'linear'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(512, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'] )
model.fit(x_train, y_train, epochs=1000, batch_size=60, validation_split=0.2)

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
5/5 [==============================] - 0s 6ms/step - loss: 24.1008 - mae: 3.8754 - mse: 24.1008 - val_loss: 6704.1328 - val_mae: 65.0462 - val_mse: 6704.1328
Epoch 1000/1000
5/5 [==============================] - 0s 6ms/step - loss: 18.1767 - mae: 3.2898 - mse: 18.1767 - val_loss: 6826.4365 - val_mae: 64.8685 - val_mse: 6826.4365
3/3 [==============================] - 0s 997us/step - loss: 4835.7036 - mae: 55.8726 - mse: 4835.7036
RMSE :  69.5392240558769
R2 :  0.2324483822589084
"""