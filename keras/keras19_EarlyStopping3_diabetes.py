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
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=500, batch_size=60, validation_split=0.2, verbose=1,
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

#5. draw, submit
import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('diabetes loss')
plt.legend(loc='center right')
plt.show()

"""
Epoch 30/500
1/5 [=====>........................] - ETA: 0s - loss: 2343.0615 - mae: 37.5444 - mse: 2343.0615Restoring model weights from the end of the best epoch: 25.
5/5 [==============================] - 0s 6ms/step - loss: 3032.5745 - mae: 44.1787 - mse: 3032.5745 - val_loss: 3038.4075 - val_mae: 43.6331 - val_mse: 3038.4075
Epoch 00030: early stopping
3/3 [==============================] - 0s 991us/step - loss: 2958.3276 - mae: 45.1336 - mse: 2958.3276
loss :  [2958.32763671875, 45.133575439453125, 2958.32763671875]
RMSE:  54.39050883570199
r2:  0.5304366912618437
"""