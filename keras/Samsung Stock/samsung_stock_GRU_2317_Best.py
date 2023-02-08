import numpy as np
import pandas as pd

#1. data
path = 'c:/study/stock/data/'
samsung = pd.read_csv(path + '삼성전자 주가.csv', index_col=0, header=0, encoding='cp949', sep=',')
amore = pd.read_csv(path + '아모레퍼시픽 주가.csv', index_col=0, header=0, encoding='cp949', sep=',')

samsung = samsung.drop(samsung.columns[[4,5,6,8,9,10,11,12,13,14,15]], axis=1)
amore = amore.drop(amore.columns[[4,5,6,8,9,10,11,12,13,14,15]], axis=1)

samsung = samsung.drop(samsung.index[range(1160,1980)], axis=0)
amore = amore.drop(amore.index[range(1160,2220)], axis=0)

# print(samsung.isnull().sum())           # NULL 값 없음.
# print(amore.isnull().sum())             # NULL 값 없음.

for i in range(len(samsung.index)):
    for j in range(len(samsung.iloc[i])):
        samsung.iloc[i,j] = int(samsung.iloc[i,j].replace(',',''))

for n in range(len(amore.index)):
    for m in range(len(amore.iloc[n])):
        amore.iloc[n,m] = int(amore.iloc[n,m].replace(',',''))

samsung = samsung.sort_values(['일자'], ascending=[True])
amore = amore.sort_values(['일자'], ascending=[True])

samsung = samsung.values
amore = amore.values

samsung = samsung.astype(np.int64)
amore = amore.astype(np.int64)

# print(type(samsung), type(amore))       # 둘 다 numpy.ndarray
# print(samsung.dtype, amore.dtype)       # 둘 다 int64

#1-1. data split

def split_xy5(dataset, time_steps, y_column):
    x, y = list(), list()
    for k in range(len(dataset)):
        x_end_number = k + time_steps
        y_end_number = x_end_number + y_column
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[k:x_end_number, :]
        tmp_y = dataset[x_end_number:y_end_number, 0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x_sam, y_sam = split_xy5(samsung, 5, 1)
x_amo, y_amo = split_xy5(amore, 5, 1)


# print(x_sam[-1,:], "\n", y_sam[-1])
# print(x_sam.shape, y_sam.shape)     # (1155, 5, 5) (1155, 1)

# print(x_amo[-1,:], "\n", y_amo[-1])
# print(x_amo.shape, y_amo.shape)     # (1155, 5, 5) (1155, 1)

def split_pred(dataset2, time_steps2, y_column2):
    x2 = list()
    for q in range(len(dataset2)):
        x_end_number2 = q + time_steps2
        y_end_number2 = x_end_number2 + y_column2
        
        if y_end_number2 > len(dataset2)+1 :
            break
        tmp_x2 = dataset2[q:x_end_number2, :]
        x2.append(tmp_x2)
    return np.array(x2)

x_sam_pred = split_pred(samsung, 5, 1)
x_amo_pred = split_pred(amore, 5, 1)

# print(x_sam_pred[-1,:])
# print(x_sam_pred.shape)

# print(x_amo_pred[-1,:])
# print(x_amo_pred.shape)


from sklearn.model_selection import train_test_split
x_sam_train, x_sam_test, y_sam_train, y_sam_test, x_amo_train, x_amo_test = train_test_split(
    x_sam, y_sam, x_amo, train_size=0.75, random_state=42, shuffle=True
)

# print(x_sam_train.shape, x_sam_test.shape, x_amo_train.shape, x_amo_test.shape) # (866, 5, 5) (289, 5, 5) (866, 5, 5) (289, 5, 5)
# print(y_sam_train.shape, y_sam_test.shape, y_amo_train.shape, y_amo_test.shape) # (866, 1) (289, 1) (866, 1) (289, 1)

x_sam_train = np.reshape(x_sam_train, (x_sam_train.shape[0], x_sam_train.shape[1]*x_sam_train.shape[2]))
x_sam_test = np.reshape(x_sam_test, (x_sam_test.shape[0], x_sam_test.shape[1]*x_sam_test.shape[2]))
x_amo_train = np.reshape(x_amo_train, (x_amo_train.shape[0], x_amo_train.shape[1]*x_amo_train.shape[2]))
x_amo_test = np.reshape(x_amo_test, (x_amo_test.shape[0], x_amo_test.shape[1]*x_amo_test.shape[2]))
x_sam_pred = np.reshape(x_sam_pred, (x_sam_pred.shape[0], x_sam_pred.shape[1]*x_sam_pred.shape[2]))
x_amo_pred = np.reshape(x_amo_pred, (x_amo_pred.shape[0], x_amo_pred.shape[1]*x_amo_pred.shape[2]))

# print(x_sam_train.shape, x_sam_test.shape, x_amo_train.shape, x_amo_test.shape) # (866, 25) (289, 25) (866, 25) (289, 25)

#1-2. scale
########### 1 StandardScaler ##################
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
scaler1.fit(x_sam_train)
x_sam_train = scaler1.transform(x_sam_train)
x_sam_test = scaler1.transform(x_sam_test)
x_sam_pred = scaler1.transform(x_sam_pred)

scaler2 = StandardScaler()
scaler2.fit(x_amo_train)
x_amo_train = scaler2.transform(x_amo_train)
x_amo_test = scaler2.transform(x_amo_test)
x_amo_pred = scaler2.transform(x_amo_pred)
################################################


########## 2 MinMaxScaler ######################
# from sklearn.preprocessing import MinMaxScaler
# scaler1 = MinMaxScaler()
# scaler1.fit(x_sam_train)
# x_sam_train = scaler1.transform(x_sam_train)
# x_sam_test = scaler1.transform(x_sam_test)
# x_sam_pred = scaler1.transform(x_sam_pred)

# scaler2 = MinMaxScaler()
# scaler2.fit(x_amo_train)
# x_amo_train = scaler2.transform(x_amo_train)
# x_amo_test = scaler2.transform(x_amo_test)
# x_amo_pred = scaler2.transform(x_amo_pred)
################################################

# print(x_sam_train[0,:], x_amo_train[0,:]) # 스케일 잘 됐는지 확인

x_sam_train = np.reshape(x_sam_train, (x_sam_train.shape[0], 5, 5))
x_sam_test = np.reshape(x_sam_test, (x_sam_test.shape[0], 5, 5))
x_amo_train = np.reshape(x_amo_train, (x_amo_train.shape[0], 5, 5))
x_amo_test = np.reshape(x_amo_test, (x_amo_test.shape[0], 5, 5))
x_sam_pred = np.reshape(x_sam_pred, (x_sam_pred.shape[0], 5, 5))
x_amo_pred = np.reshape(x_amo_pred, (x_amo_pred.shape[0], 5, 5))

#2. model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Bidirectional, Conv1D, GRU

#2-1. model_samsung
input1 = Input(shape=(5,5))
dense1 = GRU(128, activation='linear', name='sam11')(input1)
dense2 = Dense(64, activation='linear', name='sam12')(dense1)
dense3 = Dense(50, activation='linear', name='sam13')(dense2)
output1 = Dense(32, activation='linear', name='sam14')(dense3)


#2-2. model_amore
input2 = Input(shape=(5,5))
dense21 = GRU(128, activation='linear', name='amo21')(input2)
dense22 = Dense(64, activation='linear', name='amo22')(dense21)
dense23 = Dense(50, activation='linear', name='amo23')(dense22)
output2 = Dense(32, activation='linear', name='amo24')(dense23)

#2-3. model merge
from tensorflow.keras.layers import concatenate
merge1 = concatenate([output1, output2], name='mg1')
merge2 = Dense(128, activation='linear', name='mg2')(merge1)
merge3 = Dense(64, activation='linear', name='mg3')(merge2)
merge4 = Dense(50, activation='linear', name='mg4')(merge3)
merge5 = Dense(32, activation='linear', name='mg5')(merge4)
merge6 = Dense(16, activation='linear', name='mg6')(merge5)
last_output = Dense(1, activation='linear', name='last')(merge6)
model = Model(inputs=[input1, input2], outputs=last_output)

# model.summary()

#3. compile, fit, save
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

path = 'c:/study/stock/'

import datetime, time
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath_MCP = 'c:/study/stock/MCP/'
filename_MCP = '{epoch:04d}-{val_loss:.4f}.hdf5'
filepath_weights = 'c:/study/stock/weights/'

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=77, restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True, filepath = filepath_MCP + date + '_' + filename_MCP)

start = time.time()
model.fit([x_sam_train, x_amo_train], y_sam_train, validation_split=0.2, epochs=1000, batch_size=32, callbacks=[es, mcp])
end = time.time()

model.save_weights(filepath_weights + 'W_' + date + '.h5')

#4. evaluate, predict 
from sklearn.metrics import mean_squared_error, r2_score
loss, mse = model.evaluate([x_sam_test, x_amo_test], y_sam_test)

y_sam_predict = model.predict([x_sam_test, x_amo_test])

def RMSE(y_sam_test, y_sam_predict):
        return np.sqrt(mean_squared_error(y_sam_test, y_sam_predict))
r2 = r2_score(y_sam_test, y_sam_predict)

for l in range(5):
    print('시가_test : ', y_sam_test[l], '/ 예측가_test : ', y_sam_predict[l])

print('loss : ', loss)
print('mse : ', mse)
print('RMSE : ', RMSE(y_sam_test, y_sam_predict))
print('R2 : ', r2)
print('time : ', end-start)

#5. submit
filepath2 = 'c:/study/stock/data/'
y_sam_submit = model.predict([x_sam_pred, x_amo_pred])

np.savetxt(filepath2 + 'submission_' + date +'.csv', y_sam_submit[-1])

for p in range(5):
    print('시가_pred : ', y_sam[-5+p], '/ 예측가_pred : ', y_sam_submit[-6+p])

print('월요일 예측 시가 : ', y_sam_submit[-1])

"""
■ csv 파일 (삼성전자, 아모레퍼시픽 양식 동일)
  0   1    2   3    4    5    6   7     8        9       10   11   12     13       14     15     16 
일자 시가 고가 저가 종가 전일비 G 등락률 거래량 금액(백만) 신용비 개인 기관 외인(수량) 외국계 프로그램 외인비

=> index_col=0 지정한 다음에 index number 바뀜
=> 0번째 일자는 아예 index number 없어짐
=> 1번째 시가의 index가 0으로 바뀜!

■ 주식 날짜 start : 2018-05-23 (이렇게 자름)

■ 예측값을 일일이 적지 않고 바로 저장할 수 있게 numpy를 csv로 바꿔주는 함수를 사용했다.

■ 결론
이 자료에서는 relu보다 linear가, minmax보다 standard가 성능이 더 좋은 것 같다.
conv1d보다 lstm이 더 좋은 것 같고 lstm보다 gru가 성능이 더 좋은 것 같다.
dropout은 안 하는 게 성능이 더 좋은 것 같다.
"""

"""
<결과>
Epoch 398/1000
19/22 [========================>.....] - ETA: 0s - loss: 722090.1875 - mse: 722090.1875Restoring model weights from the end of the best epoch: 321.

Epoch 00398: val_loss did not improve from 420270.75000
22/22 [==============================] - 0s 12ms/step - loss: 667198.6875 - mse: 667198.6875 - val_loss: 648590.3750 - val_mse: 648590.3750     
Epoch 00398: early stopping
10/10 [==============================] - 0s 8ms/step - loss: 385396.4375 - mse: 385396.4375
시가_test :  [67600] / 예측가_test :  [67219.016]
시가_test :  [45200] / 예측가_test :  [45791.08]
시가_test :  [43050] / 예측가_test :  [42345.574]
시가_test :  [43250] / 예측가_test :  [42873.402]
시가_test :  [71000] / 예측가_test :  [71391.03]
loss :  385396.4375
mse :  385396.4375
RMSE :  620.8031505299974
R2 :  0.9977157018904285
time :  141.44066715240479
시가_pred :  [60500] / 예측가_pred :  [60341.875]
시가_pred :  [62100] / 예측가_pred :  [61506.156]
시가_pred :  [63500] / 예측가_pred :  [61875.086]
시가_pred :  [63800] / 예측가_pred :  [63682.508]
시가_pred :  [64400] / 예측가_pred :  [63852.63]
월요일 예측 시가 :  [64716.43]
"""