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
model = Sequential()                                                       # (N, 3, 1)
model.add(LSTM(units=1024, input_shape=(3,1), return_sequences=True))      # (N, 3, 1024)
model.add(LSTM(512))                                                       # (N, 512)
model.add(Dense(2048, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()

"""
#3. compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=3000)

#4. evaluate, predict
loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x_predict)
print('y_predict : ', y_predict)
"""

"""
<학습 내용>
■ 연속으로 LSTM 구성할 때 reshape 하는 방법
LSTM은 3차원 자료 줘야하는데 LSTM 거치면 2차원 자료로 바뀌므로
LSTM 연속으로 layer 추가하고 싶으면 다시 reshape 해야 한다.
reshape 하는 방법에는 2가지가 있는데,
reshape layer를 추가하거나 이전 LSTM에서 return_sequence=True로 설정하면 된다.

return_sequences=True로 하면 (N, 3, 1) 에서 앞 부분 N, 3은 그대로 내려와서
LSTM은 (N, 3, 1024)가 된다.
"""