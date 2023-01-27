import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Flatten

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


# Conv2D는 3차원 데이터 필요하기 때문에 (72,2,2) => (72,2,2,1) 로 reshape
x_train = x_train.reshape(72,2,2,1)
x_test = x_test.reshape(24,2,2,1)
x_predict = x_predict.reshape(7,2,2,1)

#2. model
model = Sequential()
model.add(Conv2D(256, (2,2), input_shape=(2,2,1), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=3000)

#4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_predict)
print('y_predict : ', y_predict)

"""
<1>
Epoch 3000/3000
3/3 [==============================] - 0s 3ms/step - loss: 7.6678e-05
1/1 [==============================] - 0s 116ms/step - loss: 4.7211e-06
loss :  4.721120603790041e-06
y_predict :  [[100.00276] [101.00279] [102.0028 ] [103.00282] [104.00283] [105.00286] [106.00287]]

<2>
Epoch 3000/3000
3/3 [==============================] - 0s 4ms/step - loss: 4.5014e-06
1/1 [==============================] - 0s 111ms/step - loss: 1.4337e-05
loss :  1.4336562344396953e-05
y_predict :  [[ 99.99779 ] [100.997795] [101.9978  ] [102.99782 ] [103.99784 ] [104.99786 ] [105.99789 ]]
 
<3>
Epoch 3000/3000
3/3 [==============================] - 0s 4ms/step - loss: 3.4364e-06
1/1 [==============================] - 0s 120ms/step - loss: 1.3839e-05
loss :  1.3838957784173544e-05
y_predict :  [[ 99.996994] [100.997025] [101.99708 ] [102.99713 ] [103.99717 ] [104.99724 ] [105.9973  ]]

★ 결론
이 자료에서는 RNN, DNN보다 CNN이 가장 좋은 것 같다.
"""