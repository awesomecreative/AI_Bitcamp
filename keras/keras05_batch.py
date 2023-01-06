import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

#2. 모델구성
model = Sequential()
model.add(Dense(3,input_dim=1))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=200, batch_size=7)

#4. 평가, 예측
result = model.predict([6])
print('6의 결과 : ', result)

"""
Epoch 200/200
1/1 [==============================] - 0s 1ms/step - loss: 0.3344
6의 결과 :  [[5.9980183]]
"""

"""
메모
# 한 덩어리 [1,2,3,4,5,6]이 한 노드에 한 번에 들어가서 y=wx+b 나옴.
# 큰 데이터를 한 번에 넣으면 연산하다가 오버플로우돼서 폭파될 수 있다.
# 60만개 1번 훈련시키는 것보다 20만개 3번 훈련시키는 것이 더 좋음.

# 배치(Batch)는 일괄적으로 처리되는 집단을 의미한다. 여기서는 [1,2,3,4,5,6]을 의미한다.
# 배치 사이즈 조절은 훈련할 때 조절하므로 fit에서 조절한다.
# batch_size=2로 하면 데이터를 2개씩 묶어서 총 3배치로 훈련한다. (1/3,2/3,3/3 뜸)
# 6개 데이터를 batch_size=4로 하면 4개, 2개 총 2배치로 훈련한다. (1/2,2/2 뜸)
# batch_size가 전체 데이터보다 클 경우에는 그냥 통째로 훈련시킨다.
# batch_size 명시 안하면 keras에서 default(기본값)은 32이다.

# <단축키>
# 하단에 넣고 싶을때 블럭 처리하고 Tab 누르기
# 띄어써진 것 한번에 없애려면 Shift+Tab 누르기
# 확대 축소는 컨트롤 누른 상태에서 + 또는 - 키 누르기
# 주석이고
"""

"""
쌍따옴표 3개의 2쌍은
여러 개의 주석을
한 번에 정리할 수 있다.
"""
'''
따옴표 3개의 2쌍은
에러 없이 여러 개의 주석을
한 번에 정리할 수 있다.
'''