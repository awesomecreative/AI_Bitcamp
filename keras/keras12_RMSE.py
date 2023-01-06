import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,21)) # (20, )
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20]) # (20, )

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size=0.7, shuffle=True, random_state=123                                                    
)

print('x_train : ', x_train)
print('x_test : ', x_test)
print('y_train : ', y_train)
print('y_test : ', y_test)

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

print("================================")
print(y_test)
print(y_predict)
print("================================")


from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
        return np.sqrt(mean_squared_error(y_test, y_predict))
    
print("RMSE : ", RMSE(y_test, y_predict))


"""
Epoch 99/100
14/14 [==============================] - 0s 620us/step - loss: 9.7283 - mae: 2.1032
Epoch 100/100
14/14 [==============================] - 0s 664us/step - loss: 9.5141 - mae: 2.1321
1/1 [==============================] - 0s 100ms/step - loss: 14.6903 - mae: 2.9718
loss :  [14.690262794494629, 2.9717676639556885] : [mse, mae]
================================
[ 9  7  5 23  8  3]
[[14.213863 ]
 [ 5.956566 ]
 [ 5.0390882]
 [16.966295 ]
 [ 8.708998 ]
 [ 7.7915215]]
================================
RMSE :  3.8327884455088443 : mse에 루트 씌운 것을 의미한다.
"""

"""
메모

# loss 값이 2개 나오는 이유: compile 해서 evalutate 할 때 loss와 metrics를 둘 다 사용했기 때문에 각각의 loss값이 나온다.
# np.sqrt는 바깥쪽에 루트를 씌우는 걸 의미한다.
# return는 돌려주라는 의미이다.
# mean_squared_error 라는 건 sklearn 라이브러리에 정리되어 있는 함수이다.
# 훈련할 때마다 RMSE 달라질 것, 이중에 가장 작은 값이 가장 좋은 가중치이므로 이것을 사용한다.
"""