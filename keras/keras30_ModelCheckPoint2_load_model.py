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

"""
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
es = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True, verbose=1)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                      filepath=path+'MCP/keras30_ModelCheckPoint1.hdf5')

model.fit(x_train, y_train, epochs=5000, batch_size=32, validation_split=0.2, callbacks=[es, mcp], verbose=1)

"""

model = load_model(path + 'MCP/keras30_ModelCheckPoint1.hdf5')

#4. 평가, 예측
mse, mae = model.evaluate(x_test, y_test)
print('mse : ', mse)
print('mae : ', mae)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)


"""
<model.save 했을 당시 결과>
Epoch 143/5000
 1/11 [=>............................] - ETA: 0s - loss: 4.4542 - mae: 1.5499Restoring model weights from the end of the best epoch: 123.

Epoch 00143: val_loss did not improve from 14.56212
11/11 [==============================] - 0s 3ms/step - loss: 4.2886 - mae: 1.5563 - val_loss: 15.6954 - val_mae: 2.3255
Epoch 00143: early stopping
4/4 [==============================] - 0s 729us/step - loss: 19.6901 - mae: 2.6352
mse :  19.690092086791992
mae :  2.6352012157440186
RMSE :  4.437352100738997
R2 :  0.7620122817273246

<load_model 한 것>
4/4 [==============================] - 0s 987us/step - loss: 19.6901 - mae: 2.6352
mse :  19.690092086791992
mae :  2.6352012157440186
RMSE :  4.437352100738997
R2 :  0.7620122817273246
"""