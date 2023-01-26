import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

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
model.add(LSTM(units=64, input_shape=(3,1)))
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
■ LSTM(Long Short-Term Memory)
차적 데이터를 효과적으로 처리하고 분석할 수 있는 순환 신경망의 일종인 장단기 메모리의 약자이다.
내부 상태를 유지하고 정보 흐름을 제어할 수 있기에 기존 RNN에서 발생할 수 있는 vanishing gradient 문제를 해결할 수 있다.

■ LSTM의 구성 요소 : 1개의 state와 3개의 gate
1 Cell state Gate = Memory cell
: maintain the long-term context of the input sequence.
: control the flow of information into and out of the cell state.
: remember information for long periods of time.
: uses a tanh activation function, which scales the input values between -1 and 1.

1 Input Gate : controls how much new information to let into the cell state
2 Forget Gate : controls how much of the previous cell state to keep.
3 Output Gate : controls how much of the cell state to output.

■ LSTM param# 계산법
LSTM param# = 4 * { output_node * ( output_node + input_dim_feature + 1 )}
4를 곱하는 이유: input, forget, output, cell state gates 이렇게 총 4개의 게이트를 계산하는 데 사용됨.
1을 더하는 이유: bias 계산하는 데 사용됨.

예시) lstm (LSTM) (None, 64) 16896
예시) 4 * { 64 * (64+1+1)}  16896

■ SimpleRNN 보다 LSTM이 더 좋다.
"""