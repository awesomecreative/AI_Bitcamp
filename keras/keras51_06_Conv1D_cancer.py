from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM, Conv1D
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

x_train = x_train.reshape(455, 30, 1)
x_test = x_test.reshape(114, 30, 1)

#2. model
model = Sequential()
model.add(Conv1D(512, 2, input_shape=(30,1)))
model.add(Flatten())  
model.add(Dense(50, activation='linear'))
model.add(Dense(40, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#3. compile, fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True, verbose=1)

# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")

# filepath = './_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
#                      filepath = filepath + 'k39_06_' + date + '_' + filename)

hist = model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, verbose=1,
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
Epoch 16/100
1/4 [======>.......................] - ETA: 0s - loss: 0.1665 - accuracy: 0.9300Restoring model weights from the end of the best epoch: 11.
4/4 [==============================] - 0s 11ms/step - loss: 0.0982 - accuracy: 0.9643 - val_loss: 0.1386 - val_accuracy: 0.9560
Epoch 00016: early stopping
4/4 [==============================] - 0s 5ms/step - loss: 0.1339 - accuracy: 0.9649
loss :  0.13388726115226746
accuracy :  0.9649122953414917
[[0.974762  ] [0.05404034] [0.9978782 ] [0.02740948] [0.35069853] [0.02955414] [0.00828665] [0.06596398] [0.99983716] [0.989097  ]]
[1 0 1 0 0 0 0 0 1 1]
accuracy_score :  0.3684210526315789
"""