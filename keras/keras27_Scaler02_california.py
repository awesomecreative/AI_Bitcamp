import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. data
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. model
model = Sequential()
model.add(Dense(16, input_dim=8, activation = 'linear'))
model.add(Dense(64, activation = 'sigmoid'))
model.add(Dense(128, activation = 'sigmoid'))
model.add(Dense(512, activation = 'sigmoid'))
model.add(Dense(64, activation = 'sigmoid'))
model.add(Dense(16, activation = 'sigmoid'))
model.add(Dense(1, activation = 'linear'))

#3. compile, fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)
hist = model.fit(x_train,y_train, epochs=1000, batch_size=50, validation_split=0.2, verbose=1,
                 callbacks=[earlyStopping])

#4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('RMSE: ', rmse)
print('r2: ', r2)

# print("=======================================")
# print(hist) # <keras.callbacks.History object at 0x00000195CE646850>
# print("=======================================")
# print(hist.history)
# print("=======================================")
# print(hist.history['loss'])
# print("=======================================")
# print(hist.history['val_loss'])
# print("=======================================")

# #5. draw, submit
# import matplotlib.pyplot as plt

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
# plt.grid()
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title('california loss')
# plt.legend(loc='center right')
# plt.show()

"""
1) scaling 안 할 때
Epoch 49/1000
237/265 [=========================>....] - ETA: 0s - loss: 0.6725 - mae: 0.6222Restoring model weights from the end of the best epoch: 39.
265/265 [==============================] - 0s 1ms/step - loss: 0.6643 - mae: 0.6182 - val_loss: 0.6499 - val_mae: 0.6291
Epoch 00049: early stopping
129/129 [==============================] - 0s 1ms/step - loss: 0.6088 - mae: 0.5738
loss :  [0.6088351011276245, 0.5737860798835754]
RMSE:  0.7802788042075763
r2:  0.5421653372710271

2) MinMaxScaler
Epoch 124/1000
248/265 [===========================>..] - ETA: 0s - loss: 0.3787 - mae: 0.4335Restoring model weights from the end of the best epoch: 114.
265/265 [==============================] - 0s 1ms/step - loss: 0.3775 - mae: 0.4330 - val_loss: 0.4189 - val_mae: 0.4352
Epoch 00124: early stopping
129/129 [==============================] - 0s 1ms/step - loss: 0.3673 - mae: 0.4321
loss :  [0.3672715425491333, 0.43206870555877686]
RMSE:  0.6060293648901626
r2:  0.7238173533887359

3) StandardScaler
Epoch 130/1000
226/265 [========================>.....] - ETA: 0s - loss: 0.2526 - mae: 0.3397Restoring model weights from the end of the best epoch: 120.
265/265 [==============================] - 0s 1ms/step - loss: 0.2524 - mae: 0.3402 - val_loss: 0.3022 - val_mae: 0.3555
Epoch 00130: early stopping
129/129 [==============================] - 0s 1ms/step - loss: 0.2590 - mae: 0.3414
loss :  [0.2590312361717224, 0.3413968086242676]
RMSE:  0.5089510855663552
r2:  0.8052124744362081

: California에선 StandardScaler가 가장 좋게 나왔다.
"""