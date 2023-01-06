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

"""
메모

# [:7]이면 0부터 시작해서 7빼기하나, 즉 6까지임. [7:]이면 0부터 시작했다고 가정하에 7부터 끝까지임.
# [:7]은 [:-3]과 같다. [7:]은 [-3:]과 같다. [:-3]은 -3빼기하나 즉 뒤에서 -1로 시작해 -4번째에 있는 값까지를 의미한다.
# Q) [7:10]이면 [7,8,9]가 아니라 [8,9,10]이 되는 이유? : 시작이 0이기 때문이다.
# print(x_train) 하면 데이터값 나옴 / 나머지 전체 주석해서 빠르게 실행하기
# 주석 2번 넣으려면 """ ''' ''' """ 쌍따옴표 바깥에, 작은 따옴표 안에 넣기!
# 같은 쌍따옴표 2번 """ """ """ """ 하면 바깥 쌍따옴표는 인식 안 됨.

# 즉, 앞에서 시작할 때는 처음에 0부터 시작하고, 뒤에서 시작할 때는 처음에 -1로 시작한다.
# 데이터 분할할 때 앞에 train, 뒤에 test로 딱 나눠서 자르면 train에서 생긴 오차가 test에 그대로 영향을 준다. 이는 데이터 양이 많아질수록 더 심해진다.
# 예를 들어, 여기서 weight가 0.999라고 해도 데이터가 1000만개 넘어가면 뒤에 오차가 점점 벌어짐. 
# 따라서 데이터 분할할 때 전체 데이터 범위내에서 train과 test를 분리함 (즉, 한 칸씩 띄어서 잡아주거나 여러칸씩 띄어서 잡아줌)
# 데이터 전체 내에서 train과 test를 분리하면 과적합 문제가 생길 수 있다.

# 통상적으로 training data를 X_train, x_train을 사용하고 가끔씩 train_x라 쓰기도 한다.
# x_train과 x_test는 각각 (7, ), (3, )이지만 열의 개수가 같기 때문에 fiture, column, 특성이 같으므로 오류가 안 난다.
"""