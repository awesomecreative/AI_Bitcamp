import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])    # (10, )
y = np.array(range(10))                 # (10, )

# 실습 : 넘파이 리스트 슬라이싱하기. 7:3으로 잘라라
x_train = x[:-3]
x_test = x[-3:]
y_train = y[:7]
y_test = y[7:]

print(x_train)      # [1 2 3 4 5 6 7]
print(x_test)       # [8 9 10]
print(y_train)      # [0 1 2 3 4 5 6]
print(y_test)       # [7 8 9]

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(10))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
result = model.predict([11])
print('[11]의 결과 : ', result)

"""
Epoch 500/500
7/7 [==============================] - 0s 831us/step - loss: 0.1027
1/1 [==============================] - 0s 112ms/step - loss: 0.0351
loss :  0.03512271121144295
[11]의 결과 :  [[9.951219]]
"""