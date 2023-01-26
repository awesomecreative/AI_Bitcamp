import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

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
model.add(SimpleRNN(64, input_shape=(3,1), activation='relu')) # 행무시 열우선
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))

model.summary()

"""
<학습 내용>
SimpleRNN의 파라미터수 계산하는 방법
total param# = units * ( units + input_dim_feature + 1 )
output = units, input_shape = (N, 3, 1) = (batch, timesteps, feature) = (N, input_length, input_dim)

예시) simple_rnn (SimpleRNN) (None, 64) 4224
예시) 64*64 + 64*1 + 64 = 4224 = 64 * (64 + 1 + 1)
"""