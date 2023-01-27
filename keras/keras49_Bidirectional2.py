# return_sequences=True 넣기!

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional

#1. data
a = np.array(range(1, 101))         # [1,2,3,4,5,6, ... ,97, 98, 99, 100]
x_predict = np.array(range(96, 106))    # [96,97,98,...,102,103,104,105]
# 예상 y = 100~106 

timesteps = 5                       # x는 4개, y는 1개

def split_x(dataset, timesteps) :
    aaa = []
    for i in range(len(dataset) - timesteps +1):
        subset = dataset[i : (i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)

a = split_x(a, timesteps)
x = a[:, :-1]
y = a[:, -1]

x_predict = split_x(x_predict, 4)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=123)


print(x_train.shape, x_test.shape)      # (72, 4) (24, 4)

x_train = x_train.reshape(72,4,1)
x_test = x_test.reshape(24,4,1)
x_predict = x_predict.reshape(7,4,1)

#2. model
model = Sequential()
# model.add(LSTM(256, input_shape=(4,1), activation='relu'))
model.add(Bidirectional(LSTM(256, activation='relu', return_sequences=True), input_shape=(4,1)))
model.add(LSTM(64))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()

#3. compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000)

#4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_predict)
print('y_predict : ', y_predict)

"""
Epoch 1000/1000
3/3 [==============================] - 0s 29ms/step - loss: 0.0159
1/1 [==============================] - 0s 430ms/step - loss: 0.0197
loss :  0.019659457728266716
y_predict :  [[ 99.45141 ] [100.16399 ] [100.81178 ] [101.398926] [101.93091 ] [102.413284] [102.85137 ]]
"""

"""
<학습 내용>
■ return_sequences
Bidirectional 자체에는 return_sequences 가 없다.
Bidirectional 안에 있는 LSTM layer에 return_sequences=True 해야 다음 LSTM layer를 추가할 수 있다.
"""