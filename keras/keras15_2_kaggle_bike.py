# https://www.kaggle.com/competition/bike-sharing-demand

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/bike/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)
print(train_csv.columns)

print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())

##### 결측치 처리 1. 제거 #####
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.shape)

x = train_csv.drop(['casual','registered','count'], axis=1)
print(x) #[10886 rows x 8 columns]
y = train_csv['count']
print(y)
print(y.shape) # (10886,)
print(test_csv) # [6493 rows x 8 columns]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=44)
print(x_train.shape, x_test.shape) # (8164, 8) (2722, 8)
print(y_train.shape, y_test.shape) # (8164,) (2722,)

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=8, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam', metrics=['mae','mse'])
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=16)
end = time.time()

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print(y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print ('RMSE : ', rmse)

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

y_submit = model.predict(test_csv)
print(y_submit)
print(y_submit.shape) # (6493, 1)
print(submission_csv.shape) # (6493, 1)

submission_csv['count'] = y_submit
print(submission_csv)

submission_csv.to_csv(path + 'submission_01061117.csv')
print ('RMSE : ', rmse)
print('R2 : ', r2)
print('time: ', end-start)

"""
202301061030
RMSE :  209.1059896152967
time:  0.5138792991638184

202301061034
RMSE :  158.64127166004116
time:  0.7845146656036377

202301061126
RMSE :  149.95034253268136
R2 :  0.3008739289964464
time:  45.58369159698486
"""