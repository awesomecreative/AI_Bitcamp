import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. data
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. model
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

"""
<연산량>
layer node 수를 1-5-4-3-2-1 로 하면
연산량은 1x5+5x4+4x3+3x2+2x1 = 45번임.
연산량은 오로지 더하기와 곱하기로만 되어 있음.
연산되는 선 하나를 파라미터라고 함.
하지만 실제 파라미터수는 기존 연산량 + bias 연산량으로
기존 연산량에 각 layer의 node수가 한 번 더 추가된다.

동그라미(가중치구하는) 네모(바이어스구하는)
5
4
3
2
1

<.summary()>
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 5)                 10        

 dense_1 (Dense)             (None, 4)                 24

 dense_2 (Dense)             (None, 3)                 15

 dense_3 (Dense)             (None, 2)                 8

 dense_4 (Dense)             (None, 1)                 3

=================================================================
Total params: 60
Trainable params: 60
Non-trainable params: 0
_________________________________________________________________

남의 것 가중치 그대로 가져와서 쓰는 전이 학습할 때 그건 훈련시킬 필요가 없음.
그때 Non-trainable params 가 나옴.
"""