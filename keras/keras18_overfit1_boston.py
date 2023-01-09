import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston

#1. data
datasets = load_boston()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=333)

#2. model
model = Sequential()
model.add(Dense(5, input_dim=13, activation = 'linear'))
model.add(Dense(5, input_shape=(13,)))
model.add(Dense(40000))
model.add(Dense(3))
model.add(Dense(20000))
model.add(Dense(1))

#3. compile, fit
import time
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae', 'mse'])
hist = model.fit(x_train, y_train, epochs=300, batch_size=1, validation_split=0.2, verbose=1)
#4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print("=======================================")
print(hist) # <keras.callbacks.History object at 0x00000195CE646850>
print("=======================================")
print(hist.history)
print("=======================================")
print(hist.history['loss'])
print("=======================================")
print(hist.history['val_loss'])
print("=======================================")

#5. submit, draw

import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('보스턴 사람수')
plt.legend(loc='center right')
plt.show()



"""
Memo
<loss, val_loss, metrics 의 dictionary>
model.fit은 훈련의 결과값을 반환하고 그걸 hist(history)라 하자.
print(hist.history)하면 loss, val_loss, metrics 등을 dictionary 형태로 보여준다.
dictionary : {'분류이름(key)' : [ , , , , ...], 'val_loss' : [ , , , , ...] (value)...} : key, value 형태이다.
print(hist.history['loss'])하면 loss 값만 볼 수 있다.

<loss, val_loss 그림 그리기>
import matplotlib.pyplot as plt : matplot library의 python plot
plt.figure(figsize=(9,6)) : figsize(가로길이, 세로길이) 단위는 inch이다.
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
: scatter : 점, plot : 직선을 표시한다.
: c는 color이고 marker는 마커표시, label은 직선의 이름(라벨)을 표시한다.
: 마커 # https://kongdols-room.tistory.com/82 # https://wikidocs.net/92083

plt.grid() : 격자 표시
plt.xlabel(' '), plt.ylabel(' '), plt.title(' ') : x축, y축, 제목 이름 표시
plt.legend(loc='upper left') : 범례표시 loc 은 locaiton(위치)를 의미한다.
loc = 'best' (default), 'upper right', 'upper left', 'upper center', 'center right', 'center left', 'center', 'lower right', 'lower center', 'lower left'
즉, upper, lower, center, right, left 로 조합하면 됨.

plt.show() : 그림을 보여준다.

<loss, val_loss를 통해 훈련이 잘 되는지 확인하기>
loss값을 참고하되 val_loss가 기준이 된다.
val_loss가 들쭉날쭉하므로 훈련이 잘 안되는 중이다.
val_loss가 최소인 지점이 최적의 weight 점이다.

<램의 용량과 연산량>
model.add(Dense(40000))
model.add(Dense(30000))
: 4만 곱하기 3만으로 연산량 약 12억이므로 중간에 메모리 부족하다고 오류 뜸.

model.add(Dense(40000))
model.add(Dense(3))
model.add(Dense(30000))
: 4만 곱하기 3, 3 곱하기 3만 이므로 최대치 약 12만으로 메모리 안 부족함.
"""

"""
<matplotlib 한글 깨짐>
# https://bskyvision.com/entry/python-matplotlibpyplot%EB%A1%9C-%EA%B7%B8%EB%9E%98%ED%94%84-%EA%B7%B8%EB%A6%B4-%EB%95%8C-%ED%95%9C%EA%B8%80-%EA%B9%A8%EC%A7%90-%EB%AC%B8%EC%A0%9C-%ED%95%B4%EA%B2%B0-%EB%B0%A9%EB%B2%95
윈도우 PC에서는 폰트가 C:\Windows\Fonts에 위치한다.\
여기서 쓰고자 하는 폰트의 속성에 들어가 폰트의 영문이름을 확인한다.
ex) 맑은 고딕 보통은 malgun.ttf 이다.

1번째 방법
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/malgun.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

2번째 방법
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False
"""