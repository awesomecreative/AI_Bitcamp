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
model.add(Bidirectional(LSTM(256, activation='relu'), input_shape=(4,1)))
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
<학습 내용>
■ Bidirectional
Bidirectional은 그 자체로 모델이 아니라 안에 layer를 넣고, input_shape을 따로 빼줘야한다.

■ 단방향 LSTM vs 양방향 Bidirectional

model.summary() 하면
lstm (LSTM)                   (None, 256)               264192
bidirectional (Bidirectional) (None, 512)               528384
연산량이 Bidirectional이 LSTM보다 딱 2배 많다는 것을 알 수 있다.

단방향 LSTM
Epoch 1000/1000
3/3 [==============================] - 0s 5ms/step - loss: 0.0064
1/1 [==============================] - 0s 175ms/step - loss: 0.0043
loss :  0.004334896802902222
y_predict :  [[100.08718 ] [101.084595] [102.08174 ] [103.07855 ] [104.07505 ] [105.07121 ] [106.06706 ]]

양방향 Bidirectional
Epoch 1000/1000
3/3 [==============================] - 0s 7ms/step - loss: 7.9213e-05
1/1 [==============================] - 0s 259ms/step - loss: 4.0655e-04
loss :  0.00040655399789102376
y_predict :  [[ 99.993065] [100.99246 ] [101.9919  ] [102.99132 ] [103.99075 ] [104.990204] [105.989685]]

★ 결과
단방향보다 양방향으로 하는 게 더 좋게 나왔다.
"""