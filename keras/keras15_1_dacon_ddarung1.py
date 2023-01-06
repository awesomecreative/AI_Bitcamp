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

x = train_csv.drop(['count'], axis=1)
print(x) # [1459 rows x 9 columns]
y = train_csv['count']
print(y)
print(y.shape) # (1459,)
print(test_csv)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=1234)
print(x_train.shape, x_test.shape) # (1021, 9) (438, 9)
print(y_train.shape, y_test.shape) # (1021,) (438,)

#2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim=9))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=10, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)
print(y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print ('RMSE : ', rmse)

r2 = r2_score(y, y_predict)
print("R2 : ", r2)

y_submit = model.predict(test_csv)

"""
메모
# panda는 데이터 분석시 사용하기 좋은 API이다. : CSV read할 때에도 쓰임 : 데이터분석쪽의 scikit-learn 같은 의미이다.
# pandas가 좋은 이유가 print하면 컬럼명과 행열수 바로 알려줌.

# 경로 . 은 현재 폴더(study)를 의미함. /는 하단 폴더를 의미함.
# train_csv = pd.read_csv('./_data/ddarung/train.csv', index_col=0)라 적을 수도 있는데 train, test, submission 앞의 경로가 같기 때문에 path로 정해준 것임.
# index_col=0 : 0번째 column은 index로 데이터가 아님을 명시해주는 것이다. (여기에서는 id), 항상 컴퓨터는 0부터 시작.
# tarin_set에 있는 마지막 count는 분리해줘야함.

# 결측치란? null 값을 의미한다. (trian_set.info())을 확인하면 non-null 값으로 null 값을 계산할 수 있다. 총 1459여야 하는데 1450만 있다면 9개가 null 값이다.
# 결측치 처리 방법: 1 결측치 데이터 삭제하기

# print(train_csv.describe()) 하면 count, mean, std, min, max 값 나옴.
# 즉, .info()는 non-null 값 알려주고 .describe는 count, mean, std, min, max 값 알려준다.

# 가장 간단한 모델 구성은 model.add(Dense(1, input_dim=x데이터의 열수)) 임.

# 평가, 예측할 때 test_csv이 아니라 x를 넣어야 함. test_csv은 subsmission을 구하기 위한 x 데이터만 있고 y 값은 없기 때문임. 즉, test_csv은 평가하기 위한 데이터가 아님.
# 즉, train은 훈련과 평가,예측를 위한 데이터이고 test는 submission 제출하기 위한 데이터임. 최종 제출은 submission만 냄.
# 단어 바꾸기 단축기 컨트롤 에프
# loss [nan, nan]으로 뜨는 이유: 데이터가 없는 란이 있기 때문이다. 데이터 없는 칸에 사칙연산을 해도 데이터가 없으므로 loss 자체를 구할 수 없다.
# 이 파일은 결측치가 나쁜 파일이다.
"""