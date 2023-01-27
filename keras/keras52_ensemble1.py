import numpy as np


#1. data
x1_datasets = np.array([range(100), range(301, 401)]).transpose()
print(x1_datasets.shape)        #(100, 2) 삼성전자 시가, 고가
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).transpose()
print(x2_datasets.shape)        #(100, 3) 아모레 시가, 고가, 종가

y = np.array(range(2001,2101)) #(100, ) 삼성전자 하루 뒤 종가


from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(
    x1_datasets, x2_datasets, y, train_size=0.7, random_state=1234
)

print(x1_train.shape, x2_train.shape, y_train.shape)    # (70, 2) (70, 3) (70,)
print(x1_test.shape, x2_test.shape, y_test.shape)       # (30, 2) (30, 3) (30,)

#2. model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. model1
input1 = Input(shape=(2,))
dense1 = Dense(11, activation='relu', name='ds11')(input1)
dense2 = Dense(12, activation='relu', name='ds12')(dense1)
dense3 = Dense(13, activation='relu', name='ds13')(dense2)
output1 = Dense(14, activation='relu', name='ds14')(dense3)

#2-2. model2
input2 = Input(shape=(3,))
dense21 = Dense(21, activation='linear', name='ds21')(input2)
dense22 = Dense(22, activation='linear', name='ds22')(dense21)
output2 = Dense(23, activation='linear', name='ds23')(dense22)

#2-3. model merge
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(12, activation='relu', name='mg2')(merge1)
merge3 = Dense(13, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

#3. compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train], y_train, epochs=100, batch_size=8)

#4. evaluate, predict
loss = model.evaluate([x1_test, x2_test], y_test)
print('loss : ', loss)

"""
■ 앙상블 (모델 여러개 merge 병합)
앙상블 학습은 여러 개의 결정 트리(Decision Tree)를 결합하여
하나의 결정 트리보다 더 좋은 성능을 내는 머신러닝 기법임.
: 앙상블은 함수형 모델로 함. => 병합도 쉽고 나누는 것도 쉬움.

예시)   삼성전자   내일 아모레퍼시픽
   시 고 저 종 거   종  시 고 저 종 거
1/1	
1/2
1/3                ◎
1/4
1/5

(3,5)로 1/3 내일 종가 ◎ 데이터 예측

삼성전자 모델과 아모레 모델을 합쳐서 삼성전자의 종가 예측
=> 삼성전자 특성과 아모레 특성 합쳐서 예측 가능

■ transpose
원래 x1_datasets.shape = (2,100)인데 transpose하면 (100,2)로 바뀜.

※ 레이어 이름을 따로 붙여주는 이유
나중에 model.summary() 할 때 컴퓨터가 임의로 이름을 지어주지 않고 내가 지정한 레이어 이름으로 보기 편하게 하려고임.

※ 항상 변수가 2개 이상이면 []로 list 만들어주기!
"""