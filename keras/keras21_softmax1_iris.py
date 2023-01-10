from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. data
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets['target']
print(x)
print(y)
print(x.shape, y.shape) # (150, 4) (150,) => input_dim=4고 마지막 Dense는 1이겠군!

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) #(150,3)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1, stratify=y)
print(y_train)
print(y_test)

#2. model
model = Sequential()
model.add(Dense(50, activation = 'relu', input_shape=(4,)))
model.add(Dense(40, activation = 'sigmoid'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(10, activation = 'linear'))
model.add(Dense(3, activation = 'softmax'))

#3. compile, fit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=1)

#4. evaluate, predict
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print('y_pred : ', y_predict)
print('y_test : ', y_test)
y_test = np.argmax(y_test, axis=1)
print('y_test : ', y_test)
acc = accuracy_score(y_test, y_predict)
print(acc)

"""
y_pred :  [0 2 1 2 2 2 0 0 0 1 2 1 0 1 0 2 0 1 1 0 1 1 2 2 0 0 2 2 1 1]
y_test :  [0 2 1 2 2 2 0 0 0 1 2 1 0 1 0 2 0 1 1 0 1 1 2 2 0 0 2 2 1 1]
acc : 1.0
"""

"""
Memo
<데이터 확인>
판다스 .descirbe()
판다스 .info()
판다스 .columns
사이킷 .DESCR
사이킷 .feature_names

<데이터 분류>
여러 데이터들중에서 너무 잘 맞아도 안되고 너무 안 맞아도 안된다.
너무 잘 맞는 데이터가 있을 경우 그 데이터에만 너무 의존하게끔 결과가 나오고,
너무 안 맞는 데이터가 있을 경우 엉뚱한 결과가 나올 수 있다.

<다중 분류: 0, 1, 2 이런식으로 3개 이상의 숫자만 나오는 경우>
<분류에서 shuffle=False의 문제점>
: shuffle=False로 하면 정렬된 데이터의 경우 y_test가 한 쪽으로 몰리게 된다.
: 예를 들어, y_train = [0 0 0 1 1 1 2 2 2], y_test = [2 2 2 2 2 2 2 2 2]
: 이렇게 되면 y_test=2인 데이터들만 테스트하게 되므로 성능이 떨어진다.
: 따라서 무조건 shuffle=True로 해야 한다.

<stratify=target : 여기에선 stratify=y>
: stratify 값을 target으로 지정해주면 각각의 class 비율(ratio)을 train, test에 유지해준다.
: 즉, 한 쪽에 쏠려서 분배되는 것을 방지한다. => 성능 차이남.
: y_train에 0,1,2가 비슷한 비율로 골고루 분포될 수 있게 하는 함수이다.
: y_test에 0,1,2가 비슷한 비율로 골고루 분포될 수 있게 하는 함수이다.

<다중분류>
model.add(Dense(원하는결과값개수, activation='softmax'))
1) 다중분류는 무조건 마지막 layer에 activation='softmax' 써야 함. 100%
2) 원하는 결과값 개수 (여기선 y의 class 개수) 적어주기.
3) compile할 때 loss='categorical_crossentropy' 적기.
3) metrics=['accuracy']

<softmax의 원리>
softmax는 입력받은 값을 출력으로 0~1사이의 값으로 모두 정규화하며 출력 값들의 총합은 항상 1이 되도록 한다.
예를 들어, 0 1 2 는 각각 1%, 49%, 50% 이렇게 나눠서 총합 100%를 만듦.

ValueError: Shapes (1, 1) and (1, 3) are incompatible

<One-Hot Encoding>
모든 데이터를 수치화하지만 1,2,3,4 모두 분류를 위한 값으로 연산이 가능한 값이 아니라
가치가 동등한 값으로 만들기 위해 One-Hot Encoding을 사용한다.
One-Hot Encoding의 원리는 값들을 좌표, 즉 벡터로 만든다는 것이다.
예시 컬럼   0   1   2       합 
        0  1   0   0      = 1
        1  0   1   0      = 1
        2  0   0   1      = 1
        모든 값을 다 합 1로 만들어 가치를 평등하게 함.
y=(150,) 에서 one-hot encoding을 거치면 y=(150,3)이 된다.
=> training하기 전에 상위 데이터셋에서 one-hot encoding 해야 함.

<One-Hot Encoding 하는 방법>
1) to_categorical
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

2) OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe.fit(y.reshape(-1,1))
y=ohe.transform(y.reshape(-1,1)).toarray()

3) get_dummies
import pandas as pd
y = pd.get_dummies(y, columns=['0','1','2']) 또는 그냥 y=pd.get_dummies(y) 라 해도 된다.
https://devuna.tistory.com/67

<y_predict의 np.argmax>
가장 마지막 node가 3개이기 때문에 y_predict도 벡터로 나옴.
0,1,2로 된 데이터가 아니라 [9.99882936e-01 1.17093179e-04 1.16871105e-20] [ ] ... 이런 식으로 원값이 나옴.
: 가장 큰 값의 위치(인덱스)를 찾기 위해서는 numpy에 있는 argmax 함수를 사용한다.
: import numpy as np
: y_predict = np.argmax(y_predict, axis=1)
: y_test가 벡터 형태이므로 np.argmax를 써서 가장 큰 값의 위치(인덱스)만 적힌 스칼라를 만든다.
: axis=1 행, axis=0 열 기준으로 계산한다는 의미임.

: 참고로 가장 작은 값의 위치(인덱스)를 찾기 위해서는 numpy에 있는 argmin 함수를 사용하면 된다.

#evaluate할 때 predict 과정 거치기 때문에 accuracy 같음.
"""