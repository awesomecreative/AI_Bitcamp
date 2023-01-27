import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#1. data

a = np.array(range(1,11))
timesteps = 5

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps)
print(bbb)

x = bbb[:, :-1]
y = bbb[:, -1]
print (x, y)                # x = [[1 2 3 4][2 3 4 5][3 4 5 6][4 5 6 7][5 6 7 8][6 7 8 9]], y = [5  6  7  8  9 10]
print (x.shape, y.shape)    # (6,4) (6, )

x_predict = np.array([7,8,9,10])
print (x_predict.shape)     # (4, )

x = x.reshape(6,4,1)
x_predict = x_predict.reshape(1,4,1)

"""
<학습 내용>
■ RNN을 위한 data split을 for문으로 하는 방법
a = np.array(range(1,6)) # [1,2,3,4,5]
timesteps = 3

def split_x(dataset, timesteps):
    aaa = []
    for i in range(len(dataset) - timesteps + 1):
        subset = dataset[i : (i + timesteps)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, timesteps)
print(bbb)

for문 3번 돌아감.
i in range(5-3+1) => i in range(3) # (0,1,2)
1 dataset[0:0+3] = [0:3] 0번째부터 시작해서 2번째까지 => [1,2,3]
2 dataset[1:1+3] = [1:4] 1번째부터 시작해서 3번째까지 => [2,3,4]
3 dataset[2:2+3] = [2:5] 2번째부터 시작해서 4번째까지 => [3,4,5]

0 1 2 3 4 번째
1 2 3 4 5 데이터

def: definition 함수를 정의함
aaa=[]: big list를 만들기 위한 단계 맨 마지막에 [] 넣어주기 위함. [[1,2,3],[2,3,4],[3,4,5]]에서 맨 바깥[]을 의미함.
for i in range(3): for문을 range안에 들어있는 숫자만큼 돌린다는 의미임. (3번 반복)
len(dataset) - timesteps + 1 : 분할되어 나오는 데이터의 개수 계산한 것.
aaa.append(subset) : subset을 aaa라는 big list에 넣는다는 의미임.

■ [:, :-1], [:, -1] 의 의미
1 2 3 4 | 5
2 3 4 5 | 6
3 4 5 6 | 7
4 5 6 7 | 8
5 6 7 8 | 9
6 7 8 9 | 10

이렇게 나누기 위해서 x = bbb[:, :-1], y = bbb[:, -1]
[:, :-1] 에서 앞의 : 는 모든 행을, 뒤의 :-1 은 마지막 열을 제외한 모든 열을 의미한다.
[:, -1] 에서 앞의 : 는 모든 행을, 뒤의 -1 은 마지막 열만을 의미한다.
"""

#2. model
model = Sequential()
model.add(LSTM(512, input_shape=(4,1), return_sequences=True))
model.add(LSTM(256))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=3000)

#4. evaluate, predict
loss = model.evaluate(x, y)
print('loss : ', loss)

y_predict = model.predict(x_predict)
print('[7,8,9,10] y_predict : ', y_predict)

"""
<결과>
<1>
Epoch 3000/3000
1/1 [==============================] - 0s 8ms/step - loss: 7.9576e-05
1/1 [==============================] - 0s 475ms/step - loss: 7.8355e-05
loss :  7.835502765374258e-05
[7,8,9,10] y_predict :  [[10.865258]]

<2>
Epoch 3000/3000
1/1 [==============================] - 0s 7ms/step - loss: 1.0935e-04
1/1 [==============================] - 1s 541ms/step - loss: 1.0848e-04
loss :  0.0001084768955479376
[7,8,9,10] y_predict :  [[10.874985]]

<3>
Epoch 3000/3000
1/1 [==============================] - 0s 7ms/step - loss: 0.0339
1/1 [==============================] - 1s 504ms/step - loss: 0.0031
loss :  0.0031027114018797874
[7,8,9,10] y_predict :  [[10.8708725]]

<4>
Epoch 3000/3000
1/1 [==============================] - 0s 7ms/step - loss: 4.7471e-05
1/1 [==============================] - 0s 472ms/step - loss: 5.2358e-05
loss :  5.235806747805327e-05
[7,8,9,10] y_predict :  [[10.888736]]

<5>
Epoch 3000/3000
1/1 [==============================] - 0s 8ms/step - loss: 0.0087
1/1 [==============================] - 0s 480ms/step - loss: 0.0141
loss :  0.014062516391277313
[7,8,9,10] y_predict :  [[11.094854]]
"""
