from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM
from sklearn.model_selection import train_test_split

#1. data
datasets = load_breast_cancer()

# print(datasets)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets['data']
y = datasets['target']
# print(x.shape, y.shape) #(569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (455, 30) (114, 30)

x_train = x_train.reshape(455, 15, 2)
x_test = x_test.reshape(114, 15, 2)

#2. model
model = Sequential()
model.add(LSTM(512, input_shape=(15,2), activation='relu'))
model.add(Dense(50, activation='linear'))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. compile, fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)

# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")

# filepath = './_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
#                      filepath = filepath + 'k39_06_' + date + '_' + filename)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.2, verbose=1,
                 callbacks=[es])
# callbacks=[es, mcp]

#4. evaluate, predict
# loss = model.evaluate(x_test, y_test)
# print('loss, accuracy : ', loss)

loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
print(y_predict[:10])   # -> 정수형으로 바꾸기!
print(y_test[:10])

import numpy as np
y_predict = y_predict.astype(int)

from sklearn.metrics import r2_score, accuracy_score
acc = accuracy_score(y_test, y_predict)
print('accuracy_score : ', acc)

"""
Epoch 27/1000
3/4 [=====================>........] - ETA: 0s - loss: 0.1509 - accuracy: 0.9333Restoring model weights from the end of the best epoch: 17.
4/4 [==============================] - 0s 49ms/step - loss: 0.1380 - accuracy: 0.9423 - val_loss: 0.2278 - val_accuracy: 0.9451
Epoch 00027: early stopping
4/4 [==============================] - 0s 9ms/step - loss: 0.1708 - accuracy: 0.9474
loss :  0.17081928253173828
accuracy :  0.9473684430122375
[[0.10908195]
 [0.245094  ]
 [0.9541633 ]
 [0.16355284]
 [0.38528675]
 [0.01697919]
 [0.01165029]
 [0.02193374]
 [0.98937166]
 [0.8841093 ]]
[1 0 1 0 0 0 0 0 1 1]
accuracy_score :  0.3684210526315789
"""