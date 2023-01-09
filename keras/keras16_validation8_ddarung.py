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

print(train_csv.isnull().sum())
train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1)

#2. 모델구성
model = Sequential()
model.add(Dense(18, input_dim=9, activation = 'linear'))
model.add(Dense(40, activation = 'relu'))
model.add(Dense(60, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))

#3. 컴파일, 훈련
import time
model.compile(loss='mse', optimizer='adam', metrics=['mae','mse'])
start = time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.2)
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
y_submit = model.predict(test_csv)

submission['count'] = y_submit
submission.to_csv(path + 'submission_val_010611.csv')
print ('RMSE : ', rmse)
print ('r2: ', r2)
print('time: ', end-start)

"""
submission_val_01060443
Epoch 200/200
54/54 [==============================] - 0s 2ms/step - loss: 2642.8577 - mae: 36.7794 - mse: 2642.8577 - val_loss: 2537.0518 - val_mae: 37.6102 - val_mse: 2537.0518
9/9 [==============================] - 0s 737us/step - loss: 2970.5466 - mae: 40.2800 - mse: 2970.5466
RMSE :  54.50272316781968
time:  18.62983274459839

submission_val_010601.csv
Epoch 200/200
54/54 [==============================] - 0s 2ms/step - loss: 2739.9653 - mae: 37.0245 - mse: 2739.9653 - val_loss: 4208.4819 - val_mae: 50.9831 - val_mse: 4208.4819
9/9 [==============================] - 0s 714us/step - loss: 3941.8743 - mae: 50.1307 - mse: 3941.8743
RMSE :  62.78434381576222
R2 :  0.46068357375169844

submission_val_010602.csv
Epoch 300/300
54/54 [==============================] - 0s 1ms/step - loss: 2567.3997 - mae: 36.4343 - mse: 2567.3997 - val_loss: 2645.7424 - val_mae: 38.9398 - val_mse: 2645.7424
9/9 [==============================] - 0s 680us/step - loss: 2842.8716 - mae: 40.5334 - mse: 2842.8716
RMSE :  53.318588236891124
r2:  0.611046026832086

submission_val_010603.csv
Epoch 300/300
54/54 [==============================] - 0s 1ms/step - loss: 2727.2720 - mae: 37.4509 - mse: 2727.2720 - val_loss: 2714.6206 - val_mae: 39.3936 - val_mse: 2714.6206
9/9 [==============================] - 0s 635us/step - loss: 2913.2566 - mae: 40.3706 - mse: 2913.2566
RMSE :  53.97459333926277
r2:  0.6014161602599131

submission_val_010604.csv
Epoch 300/300
54/54 [==============================] - 0s 1ms/step - loss: 2625.8206 - mae: 36.4577 - mse: 2625.8206 - val_loss: 2878.7688 - val_mae: 38.5312 - val_mse: 2878.7688
9/9 [==============================] - 0s 597us/step - loss: 3347.1370 - mae: 41.3958 - mse: 3347.1370
RMSE :  57.85444748745402
r2:  0.5420538316708732

submission_val_010605.csv
Epoch 500/500
54/54 [==============================] - 0s 1ms/step - loss: 1924.3502 - mae: 30.7460 - mse: 1924.3502 - val_loss: 3131.2444 - val_mae: 42.4496 - val_mse: 3131.2444
9/9 [==============================] - 0s 560us/step - loss: 3212.0076 - mae: 42.1709 - mse: 3212.0076
RMSE :  56.67457812701092
r2:  0.5605418523353086

submission_val_010606.csv
Epoch 500/500
54/54 [==============================] - 0s 1ms/step - loss: 2367.4717 - mae: 35.1832 - mse: 2367.4717 - val_loss: 2409.1584 - val_mae: 36.4766 - val_mse: 2409.1584
9/9 [==============================] - 0s 547us/step - loss: 2833.5542 - mae: 40.1462 - mse: 2833.5542
RMSE :  53.23114318761475
r2:  0.5450366615707734

submission_val_010607.csv
Epoch 500/500
54/54 [==============================] - 0s 1ms/step - loss: 2087.9001 - mae: 33.4168 - mse: 2087.9001 - val_loss: 3035.4272 - val_mae: 39.6711 - val_mse: 3035.4272
9/9 [==============================] - 0s 536us/step - loss: 2888.5366 - mae: 37.9220 - mse: 2888.5366
RMSE :  53.74510591005944
r2:  0.574328035730062

submission_val_010608.csv
Epoch 1000/1000
54/54 [==============================] - 0s 1ms/step - loss: 1399.8092 - mae: 24.7819 - mse: 1399.8092 - val_loss: 2001.2568 - val_mae: 31.0933 - val_mse: 2001.2568
9/9 [==============================] - 0s 697us/step - loss: 2128.9575 - mae: 32.4416 - mse: 2128.9575
RMSE :  46.14062707688554
r2:  0.6862641218808945

submission_val_010609.csv
Epoch 1000/1000
54/54 [==============================] - 0s 1ms/step - loss: 998.8823 - mae: 20.4561 - mse: 998.8823 - val_loss: 2231.6357 - val_mae: 33.8180 - val_mse: 2231.6357
9/9 [==============================] - 0s 623us/step - loss: 1880.8510 - mae: 29.2128 - mse: 1880.8510
RMSE :  43.3687774843936
r2:  0.6763359939389745

submission_val_010610.csv
Epoch 1000/1000
54/54 [==============================] - 0s 1ms/step - loss: 946.0626 - mae: 20.2645 - mse: 946.0626 - val_loss: 2218.9719 - val_mae: 34.0969 - val_mse: 2218.9719
9/9 [==============================] - 0s 623us/step - loss: 1760.1021 - mae: 28.8608 - mse: 1760.1021
RMSE :  41.95357010745564
r2:  0.6971149117400296

submission_val_010611.csv

훈련횟수를 증가하면 성능이 좋아진다. (여기선 1000이 괜찮은듯)
"""

"""
18-40-60-100-30-10-1, epochs=1000, batch_size=16
RMSE:  54.863184480708725
r2:  0.4820330262211937

20-50-80-100-60-40-10-1, epochs=1000, batch_size=16
RMSE:  42.8548242957806
r2:  0.683961870486326
"""