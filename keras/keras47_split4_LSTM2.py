import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

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


# feature를 2로 바꾸기 위해 reshape하기
# (b,t,f)=(batch_size, timesteps, feature) (72,4,1) => (72,2,2)로 바꾼다.
x_train = x_train.reshape(72,2,2)
x_test = x_test.reshape(24,2,2)
x_predict = x_predict.reshape(7,2,2)

#2. model
model = Sequential()
model.add(LSTM(256, input_shape=(2,2), activation='relu'))
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
3/3 [==============================] - 0s 10ms/step - loss: 7.3935e-04
1/1 [==============================] - 0s 153ms/step - loss: 3.7311e-04
loss :  0.00037310863262973726
y_predict :  [[100.065735] [101.07039 ] [102.075195] [103.08017 ] [104.08531 ] [105.09059 ] [106.09605 ]]

<2>
Epoch 3000/3000
3/3 [==============================] - 0s 11ms/step - loss: 4.5203e-04
1/1 [==============================] - 0s 145ms/step - loss: 8.1978e-04
loss :  0.0008197810384444892
y_predict :  [[100.06879 ] [101.07282 ] [102.07701 ] [103.08135 ] [104.08583 ] [105.090485] [106.09527 ]]

<3>
Epoch 3000/3000
3/3 [==============================] - 0s 10ms/step - loss: 1.2979e-04
1/1 [==============================] - 0s 168ms/step - loss: 1.5169e-04
loss :  0.00015169089601840824
y_predict :  [[100.01657 ] [101.019394] [102.02233 ] [103.02539 ] [104.02859 ] [105.031876] [106.035286]]
"""

"""
<학습 내용>
■ feature를 2로 바꾸면 좋은 점?
feature를 늘리면 자료에서 더 복잡한 관계와 패턴을 찾을 수 있어서 모델의 성능과 정확도가 올라가는 장점이 있지만,
연산량이 늘어나 과적합될 수 있을 뿐만 아니라 훈련 시간이 늘어난다는 단점이 있다.
=> 가장 좋은 균형을 찾는 것이 중요함.
"""