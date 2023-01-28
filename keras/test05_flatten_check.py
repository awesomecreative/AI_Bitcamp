 
import sklearn as sk
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense,Input,Conv1D,LSTM, Dropout, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


#!. 데이터
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
      train_size=0.7,
      shuffle=True,
      random_state=333                                                    
                                                    )

scaler = MinMaxScaler()  # train을 기준으로 삼는다
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape) # (14447, 8) (6193, 8)
x_train = x_train.reshape(14447, 8, 1, 1)
x_test = x_test.reshape(6193, 8, 1, 1)
print(x_train.shape, x_test.shape)  # (14447, 8, 1, 1)(6193, 8, 1, 1)


import datetime 
date = datetime.datetime.now()
print(date)
print(type(date)) # <class 'datetime.datetime'>  문자열 형태로 변경
date = date.strftime("%m%d_%H%M")
print(date)    # 0112_1502
print(type(date))  # <calss 'str'>

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5' 
path = './_save/'

# print(dataset.feature_names)

# print(dataset.DESCR)

# 2. 모델 구성(순차형)==============================================
model = Sequential()
model.add(Conv1D(64,2, input_shape=(8,1)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.summary()  
# =============================================================

# #2. 모델 구성 (함수형)==============================================

# input1 = Input(shape=(13,))
# dense1 = Dense(50, activation='relu')(input1)
# drop1 = Dropout(0.5)(dense1)
# dense2 = Dense(40, activation='sigmoid')(drop1)
# drop2 = Dropout(0.5)(dense2)
# dense3 = Dense(50, activation='relu')(drop2)
# drop3 = Dropout(0.5)(dense3)
# dense4 = Dense(50, activation='linear')(drop3)
# output1 = Dense(1, activation='linear')(dense4)

# model = Model(inputs=input1, outputs = output1)
# model.summary()  
# # =================================================================

# #3. 컴파일. 훈련
# import time

model.compile(loss='mse', optimizer='adam',
              metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 

es = EarlyStopping(monitor='val_loss',
                              mode='min', 
                              patience=20,
                              restore_best_weights=False,
                              verbose=1)

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1,
                      save_best_only=True,
                      filepath=filepath + 'k31_01_' + date+'_' + filename)


# # start=time.time()

model.fit(x_train, y_train, epochs=50, batch_size=256,
          validation_split=0.2,
          callbacks=[es, mcp],
          verbose=1)


# model.save(path + 'keras30_ModelCheckPoint3_save_model.h5')


# model= load_model(path + 'MCP/keras30_ModelCheckPoint1.hdf5')

#4. 평가, 예측
print('================1. 기본 출력 =================')
y_predict = model.predict(x_test, verbose=3)
loss = model.evaluate(x_test, y_test)
print('loss :',  loss)

r2 =  r2_score(y_test, y_predict)
print('R2스코어 :', r2) 