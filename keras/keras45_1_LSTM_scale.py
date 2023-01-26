import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

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
model.add(LSTM(units=1024, input_shape=(3,1)))
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
1/1 [==============================] - 0s 5ms/step - loss: 7.0897e-04
1/1 [==============================] - 0s 301ms/step - loss: 6.4425e-04
loss :  0.0006442529265768826
y_predict :  [[79.169685]]

<2>
Epoch 3000/3000
1/1 [==============================] - 0s 6ms/step - loss: 0.0781
1/1 [==============================] - 0s 298ms/step - loss: 0.0588
loss :  0.05882848799228668
y_predict :  [[79.231606]]

<3>
Epoch 3000/3000
1/1 [==============================] - 0s 7ms/step - loss: 0.0030
1/1 [==============================] - 0s 309ms/step - loss: 0.0026
loss :  0.0025966826360672712
y_predict :  [[79.20891]]
"""
