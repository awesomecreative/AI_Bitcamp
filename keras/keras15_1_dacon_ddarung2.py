import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. 데이터
path = './_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv)
print(train_csv.shape)
print(train_csv.columns)

# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')

print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())

##### 결측치 처리 1. 제거 #####
print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.shape) #(1328, 10)

x = train_csv.drop(['count'], axis=1)
print(x) # [1459 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape) # (1459,)
print(test_csv)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=44)
print(x_train.shape, x_test.shape) # (1021, 9) (438, 9)
print(y_train.shape, y_test.shape) # (929, 9) (399, 9)

#2. 모델구성
model = Sequential()
model.add(Dense(18, input_dim=9))
model.add(Dense(36))
model.add(Dense(54))
model.add(Dense(70))
model.add(Dense(90))
model.add(Dense(110))
model.add(Dense(130))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(60))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(1))

#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam', metrics=['mae','mse'])
start = time.time()
model.fit(x_train, y_train, epochs=200, batch_size=16)
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
print(y_submit.shape) # (715, 1)
print(submission.shape) # (715, 1)

# .to_csv()를 사용해서
# submission_0105.csv를 완성하시오.

submission['count'] = y_submit
print(submission)

submission.to_csv(path + 'submission_010613.csv')
print ('RMSE : ', rmse)

print('time: ', end-start)

"""
CPU 걸린 시간 : 8.462605237960815
GPU 걸린 시간 : 38.28581666946411
"""

"""
메모
# 전체 주석 달아서 중단 실행하던지 옆에 빨간점 누르고 중단점 실행하기
# print(y_submit)에서 [nan]이 나오는 이유는 test.csv도 결측치가 있었다는 뜻임.
# 하지만 test.csv에서의 있는 nan은 삭제하면 안된다. submission으로 제출해야하기때문에 공란이 있으면 안된다.

# 컬럼에 채우는 방법 : submission['count'] count란에 y_submit을 넣는다.

# .to_csv()는 panda 라이브러리에 있다. 우선 y_submit을 dataframe으로 만들고 그걸 .to_csv 한다.
# 경로는 './_data/ddarung/submission_0105.csv' 하면 된다.
# df = pd.DataFrame(y_submit)
# df.to_csv(path + 'submission_0105.csv')

# 왜 지금의 RMSE와 대회에 올라간 점수가 다를까?
# 대회에서는 내가 올린 데이터의 50%만 테스트한다. 테스트한 데이터를 public이라하고
# 나머지를 private이라 한 다음 맨 마지막 대회 최종 결과 발표날의 점수가 진짜 점수다.
# 즉, 대회에서 일부러 데이터를 왜곡한다는 의미이다.
"""