import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
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

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=60, validation_split=0.2, verbose=1,
                 callbacks=[earlyStopping])

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('RMSE: ', rmse)
print('r2: ', r2)

# #5. draw, submit
# import matplotlib.pyplot as plt

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
# plt.grid()
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title('diabetes loss')
# plt.legend(loc='center right')
# plt.show()

"""
1) scaling 안 할 때
Epoch 40/1000
1/5 [=====>........................] - ETA: 0s - loss: 2305.1692 - mae: 39.3304 - mse: 2305.1692Restoring model weights from the end of the best epoch: 30.
5/5 [==============================] - 0s 6ms/step - loss: 2868.9329 - mae: 42.9817 - mse: 2868.9329 - val_loss: 3004.8362 - val_mae: 44.2127 - val_mse: 3004.8362
Epoch 00040: early stopping
3/3 [==============================] - 0s 1ms/step - loss: 2693.2090 - mae: 42.7311 - mse: 2693.2090
loss :  [2693.208984375, 42.73112869262695, 2693.208984375]
RMSE:  51.89613331320183
r2:  0.5725179220580713

2) MinMaxScaler
Epoch 54/1000
1/5 [=====>........................] - ETA: 0s - loss: 3159.8274 - mae: 45.8546 - mse: 3159.8274Restoring model weights from the end of the best epoch: 44.
5/5 [==============================] - 0s 7ms/step - loss: 2924.5083 - mae: 43.3732 - mse: 2924.5083 - val_loss: 3070.2637 - val_mae: 45.8557 - val_mse: 3070.2637
Epoch 00054: early stopping
3/3 [==============================] - 0s 1ms/step - loss: 2698.7798 - mae: 42.4951 - mse: 2698.7798
loss :  [2698.77978515625, 42.495140075683594, 2698.77978515625]
RMSE:  51.94978190501023
r2:  0.5716336301400875

3) StandardScaler
Epoch 38/1000
1/5 [=====>........................] - ETA: 0s - loss: 2968.5376 - mae: 43.7559 - mse: 2968.5376Restoring model weights from the end of the best epoch: 28.
5/5 [==============================] - 0s 6ms/step - loss: 2587.0300 - mae: 39.8899 - mse: 2587.0300 - val_loss: 3066.6355 - val_mae: 43.4178 - val_mse: 3066.6355
Epoch 00038: early stopping
3/3 [==============================] - 0s 1ms/step - loss: 2943.2605 - mae: 42.0684 - mse: 2943.2605
loss :  [2943.260498046875, 42.06843566894531, 2943.260498046875]
RMSE:  54.25182422986222
r2:  0.5328282180538593

: Diabetes 에서는 MinMaxScaler가 가장 좋게 나왔다.
"""