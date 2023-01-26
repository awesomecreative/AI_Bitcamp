import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU

#1. data
x = np.array([[1,2,3],[2,3,4],[3,4,5],
              [4,5,6],[5,6,7],[6,7,8],
              [7,8,9],[8,9,10],[9,10,11],
              [10,11,12],[20,30,40],
              [30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])

print(x.shape, x_predict.shape) # (13, 3) (13,) (3,)

x = x.reshape(13, 3, 1)
x_predict = x_predict.reshape(1,3,1)

#2. model
model = Sequential()
model.add(GRU(units=512, input_shape=(3,1)))
model.add(Dense(2048, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=3000)

#4. evaluate, predict
loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x_predict)
print('y_predict : ', y_predict)

"""
<1>
Epoch 3000/3000
1/1 [==============================] - 0s 7ms/step - loss: 1.6253e-04
1/1 [==============================] - 0s 271ms/step - loss: 1.6200e-04
loss :  0.0001620000839466229
y_predict :  [[77.73]]

<2>
Epoch 3000/3000
1/1 [==============================] - 0s 7ms/step - loss: 2.4636e-04
1/1 [==============================] - 0s 268ms/step - loss: 2.4659e-04
loss :  0.0002465914294589311
y_predict :  [[77.7549]]

<3>
Epoch 3000/3000
1/1 [==============================] - 0s 6ms/step - loss: 8.0385e-05
1/1 [==============================] - 0s 278ms/step - loss: 7.9543e-05
loss :  7.95432788436301e-05
y_predict :  [[79.34451]]

<4>
Epoch 3000/3000
1/1 [==============================] - 0s 5ms/step - loss: 1.5355e-04
1/1 [==============================] - 0s 305ms/step - loss: 1.5299e-04
loss :  0.00015298953803721815
y_predict :  [[79.00849]]

<5>
1/1 [==============================] - 0s 6ms/step - loss: 1.3441e-04
Epoch 3000/3000
1/1 [==============================] - 0s 6ms/step - loss: 1.6669e-04
1/1 [==============================] - 0s 284ms/step - loss: 2.1129e-04
loss :  0.00021129079686943442
y_predict :  [[79.32666]]
"""
