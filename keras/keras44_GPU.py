import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU

#1. data
dataset = np.array([1,2,3,4,5,6,7,8,9,10]) # 주식 자료 11일에는? # y = ??

x = np.array([[1,2,3],
              [2,3,4],
              [3,4,5],
              [4,5,6],
              [5,6,7],
              [6,7,8],
              [7,8,9]])

y = np.array([4, 5, 6, 7, 8, 9, 10])

print(x.shape, y.shape) # (7, 3) (7,)

x = x.reshape(7,3,1)
print(x)
# [[[1],[2],[3]],[[2],[3],[4]],[[3],[4],[5]],[[4],[5],[6]],[[5],[6],[7]],[[6],[7],[8]],[[7],[8],[9]]]

print(x.shape) # (7,3,1)

#2. model
model = Sequential()
# model.add(SimpleRNN(10, input_shape=(3,1), activation='relu')) # 행무시 열우선
# model.add(SimpleRNN(units=64, input_length=3, input_dim=1, activation='relu'))
# model.add(SimpleRNN(units=64, input_dim=1, input_length=3, activation='relu'))  # => 가독성 떨어짐.
# model.add(LSTM(units=64, input_shape=(3,1)))
model.add(GRU(units=64, input_shape=(3,1)))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()

"""
<학습내용>
■ GRU
GRU는 LSTM과 유사하지만 구조가 더 단순한 RNN의 일종인 Gate Recurrent Unit을 의미한다.
GRU에는 LSTM에 존재했던 Cell이란 개념이 없다.
Update Gate : 어떤 정보를 얼만큼 유지하고 어떤 정보를 추가할 지 결정
Reset Gate : 지난 정보를 얼마나 버릴 지 조정
LSTM보다 과적합이 적고 훈련 속도가 빠르지만 LSTM 처럼 longer-term dependecies 를 처리하지 못 할 수 있다.

■ GRU param# 계산법
GRU param# = 3 * units * (units + input_dim_feature + 1)

예시) gru (GRU) (None, 64) 12864
예시) 3 * 64 * (64 + 1 + 1) = 12672
예시) 실제 12864는 이론적으로 계산해서 나온 12672와 차이가 있다.
아마 GRU param# = 3 * units * (units + input_dim_feature + 1 + 1)로
소괄호 안에 1이 하나 더 들어가는 것 같은데 그 이유는 정확히 모르겠다.

■ GRU가 LSTM보다 계산량이 적지만 LSTM 못지않은 좋은 성능을 낸다.
"""