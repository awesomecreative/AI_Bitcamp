
# [실습]
# 1. train 0.7 이상
# 2. R2 : 0.8 이상 / RMSE 사용

import sklearn as sk
print(sk.__version__) # 1.1.3

from sklearn.datasets import load_boston

# sklearn.datasets 에는 교육용 자료들이 있다.
# load_boston 에는 boston 집값이 들어가있다.
# 1번 실행하면 이 dataset을 pc에 저장하므로 다시 한 번 실행하면 속도가 빨라진다.
# scikit-learn 최신 버전이라 boston dataset이 없다면 cmd 키고 pip list에서 버전확인한다.
# pip uninstall scikit-learn==1.2.0 하고 pip install scikit-learn==1.1.3 하기.

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

print(x)
print(x.shape)      # (506, 13)
print(y)
print(y.shape)      # (506,)

print(dataset.feature_names)        # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=123)

print(dataset.DESCR)

# 터미널 우 클릭, 패널 이동해서 데이터 더 쉽게 볼 수 있음.

#2. 모델구성
model = Sequential()
model.add(Dense(13, input_dim=13))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'] )
model.fit(x_train, y_train, epochs=100, batch_size=32)

#4. 평가, 예측
model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

"""
Epoch 99/100
12/12 [==============================] - 0s 997us/step - loss: 72.7420 - mae: 6.4152
Epoch 100/100
12/12 [==============================] - 0s 981us/step - loss: 70.8750 - mae: 6.5556
5/5 [==============================] - 0s 823us/step - loss: 49.1980 - mae: 5.0940
RMSE :  7.014128886610337
R2 :  0.39132542882045973
"""
