import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])    # (10, )
y = np.array(range(10))                 # (10, )

# [검색] train과 test를 섞어서 7:3으로 만들기!
# 힌트 : 사이킷런

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    test_size=0.3,
    shuffle=True,
    random_state=123
)
 
print('x_train : ', x_train)
print('x_test : ', x_test)       
print('y_train : ', y_train)      
print('y_test : ', y_test)       

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
7/7 [==============================] - 0s 669us/step - loss: 0.1398
1/1 [==============================] - 0s 164ms/step - loss: 0.0551
loss :  0.05514019727706909
[11]의 결과 :  [[9.853865]]
"""

"""
메모

# () 괄호 안에 있는 것이 parameter임. train_size=0.7, test_size=0.4로 쓸 수 없음. The sum of test_size and train_size should be in the (0, 1) range.
# train과 test를 섞어서 7:3으로 만드는 법
# from sklearn.model_selection import train_test_split 을 입력해 train_test_split를 불러온다.
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
# train_size=0.7의 의미는 train set이 70 비율이라는 의미이다.
# test_size=0.25의 의미는 test set이 25% 비율이라는 의미이다. 즉 데이터를 75%:25%로 나눈 것이다. default는 0.25이다.
# shuffle은 데이터를 분할하기 전에 미리 섞어놓는 것을 의미한다. default는 true이다.
# random_state를 지정해줘야 여러번 실행했을 때에도 고정된 값을 얻을 수 있다. default 값은 none이다. none의 경우 여러번 실행하면 다 다른 값이 나온다. (난수값이 다 다르기 때문이다.)
# 랜덤한 값은 실제로 랜덤한 값이 아니다. 랜덤 사이즈를 지정해주면 난수표에 맞춰서 그 형태로만 나온다.

# 통상적으로 training data를 X_train, x_train을 사용하고 가끔씩 train_x라 쓰기도 한다.
# x_train과 x_test는 각각 (7, ), (3, )이지만 열의 개수가 같기 때문에 fiture, column, 특성이 같으므로 오류가 안 난다.
"""
