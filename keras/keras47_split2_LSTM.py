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

x_train = x_train.reshape(72,4,1)
x_test = x_test.reshape(24,4,1)
x_predict = x_predict.reshape(7,4,1)

#2. model
model = Sequential()
model.add(LSTM(256, input_shape=(4,1), activation='relu'))
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
3/3 [==============================] - 0s 5ms/step - loss: 1.6625e-04
3/3 [==============================] - 0s 2ms/step - loss: 1.2584e-04
loss :  0.00012584160140249878
y_predict :  [[ 99.96789 ] [100.927635] [101.87261 ] [102.80047 ] [103.70882 ] [104.59539 ] [105.458015]]

<2>
Epoch 3000/3000
3/3 [==============================] - 0s 10ms/step - loss: 0.0037
3/3 [==============================] - 0s 0s/step - loss: 0.0022 
loss :  0.0022174590267241
y_predict :  [[ 99.95273 ] [100.955795] [101.95902 ] [102.96244 ] [103.96601 ] [104.96976 ] [105.97364 ]]

<3>
Epoch 3000/3000
3/3 [==============================] - 0s 9ms/step - loss: 1.1434e-04
3/3 [==============================] - 0s 0s/step - loss: 6.1608e-05
loss :  6.160790508147329e-05
y_predict :  [[100.029945] [101.034294] [102.03889 ] [103.043655] [104.04866 ] [105.05387 ] [106.05928 ]]

<4>
Epoch 3000/3000
3/3 [==============================] - 0s 8ms/step - loss: 3.8287e-05
3/3 [==============================] - 0s 8ms/step - loss: 5.2850e-05
loss :  5.284995131660253e-05
y_predict :  [[100.02438 ] [101.02718 ] [102.03014 ] [103.03324 ] [104.036476] [105.03984 ] [106.043365]]
 
<5>
Epoch 3000/3000
3/3 [==============================] - 0s 9ms/step - loss: 5.7054e-04
3/3 [==============================] - 0s 8ms/step - loss: 9.1172e-04
loss :  0.0009117207955569029
y_predict :  [[ 99.988335] [100.992195] [101.99621 ] [103.00041 ] [104.00475 ] [105.00927 ] [106.01392 ]]
"""

"""
<학습 내용>
■ 시계열 자료
: 특징 y가 없음. 직접 지정해줘야 함.

ex) 주가
주가 | 1   2   3   4   5   6   | 7◎  8   9   10
환율 | 100 101 102 103 104 105 | 106 107 108 109
원유 | 100 100 1   2   3   99  | 50  55  100 123
금리 | 3.5 3.1 3.0 1.1 2.1 1.2 | 1.3

x, y 블럭으로 지정 가능
3x3 or 4x4 or 3x1 or ...
주가 환율 원유의 가격으로 그 다음날 & 모레의 주가 예측
주가 환율 원유의 가격으로 그 다음날 주가 환율 원유 모두 예측
등 다양하게 x, y 지정 가능

위의 예시에서 주가 1~6으로 쭉 자르고 주가 7 예측할 때
(batch_size, timesteps, feature)
(N, 6, 1)
(N, 3, 2) 3x2 =6

1~6까지 6개 기준으로 자르니까 기본적인 timesteps 6,
feature는 1이면 1로 2예측 / 2로 3예측 / 3으로 4예측 / ...
만약 feature가 2라면 1,2로 3,4예측 / 3,4로 5,6예측 / 5,6으로 7,8예측 / ...
인데 feature가 2인 경우는 timesteps가 3이 된다.
(2개씩 하나, 2개씩 하나, 2개씩 하나 이렇게 3번 이동해야 7을 구할 수 있기 때문에 timesteps가 3이 된다.)

■ 시계열에서 x_predict도 split 해야할 때
x_predict는 y값을 따로 나누지 않고 모든 x_predict를 x값으로만 사용하기 때문에 timesteps만 조절해주면 된다.
굳이 [:,:-1], [:, -1] 하지 않아도 됨. => 이렇게 되면 (6,4) 나옴.
=> 맨 마지막 [102,103,104,105]가 안나오기 때문에 (7,4)로 해야함.
그렇기 때문에 바로 timesteps만 이용해 잘라주면 된다.

■ 왜 굳이 x_predict를 range(96,106)으로 주셨을까?
선생님이 굳이 x_predict = [102,103,104,105]로 주시지 않고 range(96,106)으로 주신 이유는
[96,97,98,99]의 값이 100인 걸 이미 알고 있지만 모델의 결과값도 100으로 나오는 지 확인하기 위해서이다.
궁금한 건 맨 마지막 결과값이겠지만 앞서 나온 결과값도 알기 위해서이다.

※ train_size=0.75 (default)
※ range() function takes 3 arguments
: start, stop, and step
: start means starting point
: stop means ending point
: step means interval of the numbers in the sequence
: the last number is not included.
"""