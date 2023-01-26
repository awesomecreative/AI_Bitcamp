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

# x = x.reshape(7,3,1)
# print(x)
# # [[[1],[2],[3]],[[2],[3],[4]],[[3],[4],[5]],[[4],[5],[6]],[[5],[6],[7]],[[6],[7],[8]],[[7],[8],[9]]]

# print(x.shape) # (7,3,1)

#2. model
model = Sequential()
# model.add(SimpleRNN(64, input_shape=(3,1), activation='relu')) # 행무시 열우선
model.add(Dense(128, input_shape=(3, ), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))


#3. compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000)

#4. evaluate, predict
loss = model.evaluate(x, y)
print('loss : ', loss)

y_pred = np.array([8,9,10]).reshape(1,3,1)
result = model.predict(y_pred)
print('[8,9,10]의 결과 : ', result)

"""
Epoch 1000/1000
1/1 [==============================] - 0s 9ms/step - loss: 2.1901e-07
1/1 [==============================] - 0s 142ms/step - loss: 2.1231e-07
loss :  2.1230577829101094e-07
[8,9,10]의 결과 :  [[11.032604]]
"""

"""
<학습 내용>
predict에서의 input_shape도 model.add에 들어가는 input_shape과 동일해야하며 행무시 열우선이므로 앞에 1 추가해야한다.
"""