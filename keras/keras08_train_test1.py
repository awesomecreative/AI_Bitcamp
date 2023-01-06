import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])    # (10, )
# y = np.array(range(10))                 # (10, )
x_train = np.array([1,2,3,4,5,6,7])     # (7, )
x_test = np.array([8,9,10])             # (3, )
y_train = np.array(range(7))            # (7, )
y_test = np.array(range(7,10))          # (3, )

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
Epoch 499/500
7/7 [==============================] - 0s 889us/step - loss: 0.2324
Epoch 500/500
7/7 [==============================] - 0s 756us/step - loss: 0.2054
1/1 [==============================] - 0s 103ms/step - loss: 0.6279
loss :  0.6278548240661621
[11]의 결과 :  [[10.780885]]
"""

"""
메모

# 통상적으로 training data를 X_train, x_train을 사용하고 가끔씩 train_x라 쓰기도 한다.
# x_train과 x_test는 각각 (7, ), (3, )이지만 열의 개수가 같기 때문에 fiture, column, 특성이 같으므로 오류가 안 난다.
"""