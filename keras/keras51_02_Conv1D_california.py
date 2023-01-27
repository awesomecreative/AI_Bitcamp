import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Dropout, LSTM, Conv1D
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

print(x_train.shape, x_test.shape) # (16512, 8) (4128, 8)

x_train = x_train.reshape(16512, 8, 1)
x_test = x_test.reshape(4128, 8, 1)


#2. model
model = Sequential()
model.add(Conv1D(512, 2, input_shape=(8,1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'sigmoid'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

model.summary()

#3. compile, fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)


# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")

# filepath = './_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
#                      filepath = filepath + 'k31_02_' + date + '_' + filename)

hist = model.fit(x_train,y_train, epochs=100, batch_size=50, validation_split=0.2, verbose=1,
                 callbacks=[es])
# callbacks=[es,mcp]


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
Epoch 33/100
258/265 [============================>.] - ETA: 0s - loss: 0.3317 - mae: 0.4045Restoring model weights from the end of the best epoch: 23.
265/265 [==============================] - 1s 5ms/step - loss: 0.3325 - mae: 0.4053 - val_loss: 0.3286 - val_mae: 0.3873
Epoch 00033: early stopping
129/129 [==============================] - 0s 3ms/step - loss: 0.2832 - mae: 0.3663
loss :  [0.2832019329071045, 0.3663058578968048]
RMSE:  0.5321672023768578
r2:  0.7870364580291441
"""