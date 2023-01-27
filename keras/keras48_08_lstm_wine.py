import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM

#1. data
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (178,13) (178,)
print(y) # 다중분류임을 알 수 있음.
print(np.unique(y)) # [0 1 2]
print(np.unique(y, return_counts=True)) # [0 1 2] [59 71 48]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (142, 13) (36, 13)

x_train = x_train.reshape(142, 13, 1)
x_test = x_test.reshape(36, 13, 1)

#2. model
model = Sequential()
model.add(LSTM(512, input_shape=(13,1), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(70, activation='sigmoid'))
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(80, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(3, activation='softmax'))


#3. compile, fit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)

# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")

# filepath = './_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
#                      filepath = filepath + 'k39_08_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, verbose=1, callbacks=[es])
# callbacks = [es, mcp]

#4. evaluate, predict
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
print(y_predict)

y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
print('y_predict : ', y_predict)
print('y_test : ', y_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

"""
Epoch 31/1000
112/113 [============================>.] - ETA: 0s - loss: 0.4468 - accuracy: 0.8839Restoring model weights from the end of the best epoch: 21.
113/113 [==============================] - 3s 22ms/step - loss: 0.4429 - accuracy: 0.8850 - val_loss: 0.2867 - val_accuracy: 0.8966
Epoch 00031: early stopping
2/2 [==============================] - 0s 0s/step - loss: 0.4333 - accuracy: 0.8333
loss :  0.4333137273788452
accuracy :  0.8333333134651184
[[0.16736    0.20517299 0.62746704]
 [0.01452885 0.9153929  0.07007821]
 [0.9739242  0.01331859 0.01275721]
 [0.02099966 0.17599583 0.80300456]
 [0.97174275 0.01430928 0.01394807]
 [0.01452885 0.9153929  0.07007821]
 [0.9561187  0.02109346 0.02278779]
 [0.01452885 0.9153929  0.07007821]
 [0.95577157 0.0212168  0.02301169]
 [0.01452885 0.9153929  0.07007821]
 [0.02287559 0.17831737 0.798807  ]
 [0.01452885 0.9153929  0.07007821]
 [0.01452885 0.9153929  0.07007821]
 [0.994909   0.0030502  0.00204078]
 [0.01452885 0.9153929  0.07007821]
 [0.01452885 0.9153929  0.07007821]
 [0.01452885 0.9153929  0.07007821]
 [0.02031624 0.1753895  0.8042942 ]
 [0.01452885 0.9153929  0.07007821]
 [0.02105761 0.17603774 0.80290467]
 [0.04179769 0.18789893 0.7703034 ]
 [0.02025302 0.17593153 0.8038155 ]
 [0.9959941  0.00244649 0.00155944]
 [0.01955283 0.17617664 0.8042705 ]
 [0.02162335 0.17698096 0.8013957 ]
 [0.01452885 0.9153929  0.07007821]
 [0.01452885 0.9153929  0.07007821]
 [0.974631   0.01300193 0.01236711]
 [0.01452885 0.9153929  0.07007821]
 [0.02194035 0.1771489  0.8009108 ]
 [0.9905911  0.00535561 0.00405327]
 [0.9748761  0.01290678 0.01221718]
 [0.8904339  0.04651742 0.06304862]
 [0.12728967 0.20312451 0.6695858 ]
 [0.8413681  0.06356248 0.09506938]
 [0.02048883 0.17586595 0.80364525]]
y_predict :  [2 1 0 2 0 1 0 1 0 1 2 1 1 0 1 1 1 2 1 2 2 2 0 2 2 1 1 0 1 2 0 0 0 2 0 2]
y_test :  [1 1 0 2 0 1 0 1 0 1 2 1 1 0 1 1 1 2 2 2 1 2 0 2 1 2 1 0 1 2 0 0 0 0 0 2]
acc :  0.8333333333333334
"""