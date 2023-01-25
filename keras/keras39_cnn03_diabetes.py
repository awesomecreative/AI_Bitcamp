import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
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
scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (353, 10) (89, 10)

x_train = x_train.reshape(353, 5, 2, 1)
x_test = x_test.reshape(89, 5, 2, 1)

#2. 모델구성
model = Sequential()
model.add(Conv2D(512, (2,2), input_shape=(5,2,1), padding='same', activation='relu'))
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse'] )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)


import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                     filepath = filepath + 'k31_03_' + date + '_' + filename)


hist = model.fit(x_train, y_train, epochs=1000, batch_size=60, validation_split=0.2, verbose=1,
                 callbacks=[es, mcp])

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
Epoch 83/1000
1/5 [=====>........................] - ETA: 0s - loss: 2661.8381 - mae: 41.6463 - mse: 2661.8381Restoring model weights from the end of the best epoch: 73.

Epoch 00083: val_loss did not improve from 2888.34497
5/5 [==============================] - 0s 14ms/step - loss: 3130.0107 - mae: 44.6206 - mse: 3130.0107 - val_loss: 2993.3813 - val_mae: 44.3719 - val_mse: 2993.3813
Epoch 00083: early stopping
3/3 [==============================] - 0s 56ms/step - loss: 2981.3784 - mae: 43.0902 - mse: 2981.3784
loss :  [2981.37841796875, 43.09015655517578, 2981.37841796875]
RMSE:  54.60200129877472
r2:  0.5267778846064115
"""