import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler

path = './_save/'
# path = '../_save/'
# path = 'c:/study/_save/'

#1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)

# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델구성(순차형)
# model = Sequential()
# model.add(Dense(50, input_dim=13, activation = 'relu'))
# model.add(Dense(40, activation = 'sigmoid'))
# model.add(Dense(30, activation = 'relu'))
# model.add(Dense(20, activation = 'linear'))
# model.add(Dense(1, activation = 'linear'))
# model.summary()

#2. 모델구성(함수형)
input1 = Input(shape=(13,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='sigmoid')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(1, activation='linear')(dense4)
model = Model(inputs=input1, outputs=output1)
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'] )

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, 
                 # restore_best_weights=False,
                   verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                      filepath= path + 'MCP/keras30_ModelCheckPoint3.hdf5')

model.fit(x_train, y_train, epochs=5000, batch_size=32, validation_split=0.2, callbacks=[es, mcp], verbose=1)

model.save(path + 'keras30_ModelCheckPoint3_save_model.h5')

#4. 평가, 예측
print("=================1. 기본출력 ==================")
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

print("=================2. load_model 출력 ==================")
model2 = load_model(path + 'keras30_ModelCheckPoint3_save_model.h5')
mse, mae = model2.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model2.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

print("=================3. ModelCheckPoint 출력 ==================")
model3 = load_model(path + 'MCP/keras30_ModelCheckPoint3.hdf5')
mse, mae = model3.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model3.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

"""
<EarlyStopping 에서 restore_best_weights=True>
=================1. 기본 출력 ==================
4/4 [==============================] - 0s 935us/step - loss: 18.7948 - mae: 2.5197
mse :  18.794769287109375
mae :  2.519662857055664
RMSE :  4.335293695477115
R2 :  0.7728337486337346
=================2. load_model 출력 ==================
4/4 [==============================] - 0s 0s/step - loss: 18.7948 - mae: 2.5197
mse :  18.794769287109375
mae :  2.519662857055664
RMSE :  4.335293695477115
R2 :  0.7728337486337346
=================3. ModelCheckPoint 출력 ==================
4/4 [==============================] - 0s 0s/step - loss: 18.7948 - mae: 2.5197
mse :  18.794769287109375
mae :  2.519662857055664
RMSE :  4.335293695477115
R2 :  0.7728337486337346

셋 다 모두 동일하게 나온다.
"""

"""
<EarlyStopping 에서 restore_best_weights=False(Default)>
=================1. 기본 출력 ==================
4/4 [==============================] - 0s 673us/step - loss: 19.4476 - mae: 2.7759
mse :  19.44761085510254
mae :  2.775892972946167
RMSE :  4.409944196458912
R2 :  0.7649431288398971
=================2. load_model 출력 ==================
4/4 [==============================] - 0s 5ms/step - loss: 19.4476 - mae: 2.7759
mse :  19.44761085510254
mae :  2.775892972946167
RMSE :  4.409944196458912
R2 :  0.7649431288398971
=================3. ModelCheckPoint 출력 ==================
4/4 [==============================] - 0s 0s/step - loss: 17.8327 - mae: 2.5087
mse :  17.832704544067383
mae :  2.508680582046509
RMSE :  4.222878284165551
R2 :  0.7844619789871125

이론상으로 restore_best_weights=False(Default) 하면
ModelCheckPoint가 기본출력이랑 load_model한 것보다 더 좋게 나온다.
왜냐하면, restore_best_weights=False(Default) 했을 때,
ModelCheckPoint는 오차가 최소인 지점에서 브레이크 건 상태에서 결과값을 내 모델과 가중치를 저장하고,
그냥 .save는 오차가 최소인 지점에서 브레이크를 걸지 않고 patience 만큼 더 간 상태에서 결과값을 내 모델과 가중치를 저장하기 때문이다.
만약 restore_best_weights=True로 설정해주면 ModelCheckPoint랑 .save 둘 다 최소인 지점에서 브레이크를 걸기 때문에 동일한 값이 나온다.

가끔 restore_best_weights=False 해도 ModelCheckPoint의 결과가 안 좋을 때에도 있다.
그 이유는 평가하는 데이터가 x_train이 아니라 x_test이기 때문이다.
x_train 에서는 브레이크 걸었을 때 가중치가 안 걸었을 때보다 더 좋게 나오지만
x_test 에서는 브레이크 걸었을 때 가중치가 안 걸었을 때보다 안 좋을 수 있기 때문이다.
"""

