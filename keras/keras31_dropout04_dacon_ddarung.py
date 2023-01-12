import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. data
# path = './_data/ddarung/'
path = 'c:/study/_data/ddarung/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'submission.csv', index_col=0)

print(train_csv.isnull().sum())
train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

# #2. model
# model = Sequential()
# model.add(Dense(20, input_dim=9, activation = 'linear'))
# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(80, activation = 'relu'))
# model.add(Dense(100, activation = 'relu'))
# model.add(Dropout(0.5))
# model.add(Dense(60, activation = 'relu'))
# model.add(Dense(40, activation = 'relu'))
# model.add(Dense(10, activation = 'relu'))
# model.add(Dense(1, activation = 'linear'))

#2. model
input1 = Input(shape=(9,))
dense1 = Dense(20, activation='linear')(input1)
dense2 = Dense(50, activation='relu')(dense1)
dense3 = Dense(80, activation='relu')(dense2)
dense4 = Dense(100, activation='relu')(dense3)
drop4 = Dropout(0.5)(dense4)
dense5 = Dense(60, activation='relu')(drop4)
dense6 = Dense(40, activation='relu')(dense5)
dense7 = Dense(10, activation='relu')(dense6)
output1 = Dense(1, activation='linear')(dense7)
model = Model(inputs=input1, outputs=output1)

#3. compile, fit : earlystop, time
model.compile(loss='mse', optimizer='adam', metrics=['mae','mse'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=300, verbose=1, restore_best_weights=True)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                     filepath = filepath + 'k31_04_' + date + '_' + filename)

import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=10000, batch_size=16, validation_split=0.2, verbose=1,
                 callbacks=[es, mcp])
end = time.time()

#4. evaluate, predict : RMSE, r2
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('RMSE: ', rmse)
print('r2: ', r2)

#5. draw, submit

# import matplotlib.pyplot as plt

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
# plt.grid()
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title('ddarung loss')
# plt.legend(loc='center right')
# plt.show()

y_submit = model.predict(test_csv)

submission['count'] = y_submit

filepath2 = 'c:/study/_data/ddarung/'
submission.to_csv(filepath2 + 'submission_save_04_' + date + '.csv')

print('time: ', end-start)
