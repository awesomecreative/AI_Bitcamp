import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

print(x, x.shape, type(x))
print(np.min(x), np.max(x)) # 0.0 711.0

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[earlyStopping])

#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

"""
1) scaling 안 했을때
Epoch 20/1000
 1/11 [=>............................] - ETA: 0s - loss: 101.9955 - mae: 7.9430Restoring model weights from the end of the best epoch: 10.
11/11 [==============================] - 0s 3ms/step - loss: 52.4489 - mae: 5.1937 - val_loss: 90.5479 - val_mae: 6.5641
Epoch 00020: early stopping
4/4 [==============================] - 0s 918us/step - loss: 65.7731 - mae: 5.3767
mse :  65.77306365966797
mae :  5.376704692840576
RMSE :  8.110059202939055
R2 :  0.20502254563760325

2) MinMaxScaler
Epoch 42/1000
 1/11 [=>............................] - ETA: 0s - loss: 8.2863 - mae: 2.2660Restoring model weights from the end of the best epoch: 32.
11/11 [==============================] - 0s 3ms/step - loss: 9.5620 - mae: 2.3357 - val_loss: 18.7846 - val_mae: 2.7091
Epoch 00042: early stopping
4/4 [==============================] - 0s 674us/step - loss: 17.3385 - mae: 2.9480
mse :  17.3384952545166
mae :  2.9479918479919434
RMSE :  4.163951941027888
R2 :  0.790435275935063

3) StandardScaler
Epoch 77/1000
 1/11 [=>............................] - ETA: 0s - loss: 3.0748 - mae: 1.4836Restoring model weights from the end of the best epoch: 67.
11/11 [==============================] - 0s 3ms/step - loss: 3.0258 - mae: 1.2843 - val_loss: 14.2199 - val_mae: 2.2155
Epoch 00077: early stopping
4/4 [==============================] - 0s 665us/step - loss: 28.0503 - mae: 2.6601
mse :  28.050336837768555
mae :  2.660092830657959
RMSE :  5.2962569436134554
R2 :  0.6609647491524018

: Boston에서는 MinMaxScaler가 가장 좋게 나왔다.
"""