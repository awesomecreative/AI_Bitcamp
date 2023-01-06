
# [실습]
# 1. train 0.7 이상
# 2. R2 : 0.8 이상 / RMSE 사용

import sklearn as sk
print(sk.__version__) # 1.1.3

from sklearn.datasets import load_boston

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

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.75, shuffle=True, random_state=123)

print(dataset.DESCR)

#2. 모델구성
model = Sequential()
model.add(Dense(16, input_dim=13))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(512))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'] )
model.fit(x_train, y_train, epochs=500, batch_size=32)

#4. 평가, 예측
model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

"""
Epoch 499/500
12/12 [==============================] - 0s 1ms/step - loss: 35.9799 - mae: 4.5261
Epoch 500/500
12/12 [==============================] - 0s 1ms/step - loss: 36.7104 - mae: 4.4553
4/4 [==============================] - 0s 5ms/step - loss: 27.5210 - mae: 3.9301
RMSE :  5.2460436951974
R2 :  0.6513783358835672
"""

"""
메모

# sklearn.datasets 에는 교육용 자료들이 있다.
# load_boston 에는 boston 집값이 들어가있다.
# 1번 실행하면 이 dataset을 pc에 저장하므로 다시 한 번 실행하면 속도가 빨라진다.
# scikit-learn 최신 버전이라 boston dataset이 없다면 cmd 키고 pip list에서 버전확인한다.
# pip uninstall scikit-learn==1.2.0 하고 pip install scikit-learn==1.1.3 하기.

# 터미널 우 클릭, 패널 이동해서 대량의 데이터를 볼 때 더 쉽게 볼 수 있음.
"""
