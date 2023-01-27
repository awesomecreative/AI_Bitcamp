import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM, Conv1D

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
model.add(Conv1D(512, 2, input_shape=(13,1)))
model.add(Flatten())
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
Epoch 67/1000
107/113 [===========================>..] - ETA: 0s - loss: 4.8352e-07 - accuracy: 1.0000Restoring model weights from the end of the best epoch: 57.
113/113 [==============================] - 1s 5ms/step - loss: 4.5890e-07 - accuracy: 1.0000 - val_loss: 6.6181e-07 - val_accuracy: 1.0000
Epoch 00067: early stopping
2/2 [==============================] - 0s 58ms/step - loss: 0.4547 - accuracy: 0.9444
loss :  0.4546564817428589
accuracy :  0.9444444179534912
[[2.2604512e-09 1.0000000e+00 2.8095451e-10]
 [1.9537728e-10 1.0000000e+00 3.5837142e-11]
 [1.0000000e+00 3.2454944e-10 5.6212474e-10]
 [8.3334027e-11 1.9512665e-10 1.0000000e+00]
 [1.0000000e+00 4.4728509e-11 1.9542835e-10]
 [2.0600285e-10 1.0000000e+00 3.6248306e-11]
 [1.0000000e+00 1.6537945e-12 2.1107588e-11]
 [1.6523271e-07 9.9999762e-01 2.2131496e-06]
 [1.0000000e+00 8.7605201e-12 8.6447322e-11]
 [2.2568958e-10 1.0000000e+00 4.1703727e-11]
 [8.4501528e-10 2.6644429e-09 1.0000000e+00]
 [2.0782816e-10 1.0000000e+00 4.3373451e-11]
 [1.0370197e-10 1.0000000e+00 1.8424137e-11]
 [1.0000000e+00 1.4037981e-12 2.0672226e-11]
 [1.3828036e-09 1.0000000e+00 4.6085125e-10]
 [1.4190284e-10 1.0000000e+00 2.7923998e-11]
 [2.2430643e-10 1.0000000e+00 4.4455578e-11]
 [5.3918753e-10 1.7745495e-09 1.0000000e+00]
 [3.3274755e-10 1.1165037e-09 1.0000000e+00]
 [6.3307792e-10 1.6498242e-09 1.0000000e+00]
 [9.8147712e-10 1.0000000e+00 1.6533074e-10]
 [5.3579013e-10 3.0179312e-09 1.0000000e+00]
 [1.0000000e+00 1.7645025e-12 2.3579892e-11]
 [9.3244273e-11 2.5938687e-10 1.0000000e+00]
 [1.3018917e-08 1.7688434e-07 9.9999976e-01]
 [2.7212815e-04 5.5923593e-01 4.4049191e-01]
 [5.3696153e-10 1.0000000e+00 1.0769486e-10]
 [1.0000000e+00 2.3597832e-12 3.3085073e-11]
 [1.2437568e-10 1.0000000e+00 2.4510791e-11]
 [8.1983233e-11 2.2740977e-10 1.0000000e+00]
 [1.0000000e+00 2.3383129e-12 2.7824101e-11]
 [1.0000000e+00 1.9797880e-10 3.9699077e-10]
 [1.0000000e+00 1.0709463e-11 9.1459604e-11]
 [1.0000000e+00 7.9121580e-09 8.6069329e-09]
 [1.0000000e+00 2.3504265e-12 3.2323959e-11]
 [2.5029170e-09 3.2386708e-08 1.0000000e+00]]
y_predict :  [1 1 0 2 0 1 0 1 0 1 2 1 1 0 1 1 1 2 2 2 1 2 0 2 2 1 1 0 1 2 0 0 0 0 0 2]
y_test :  [1 1 0 2 0 1 0 1 0 1 2 1 1 0 1 1 1 2 2 2 1 2 0 2 1 2 1 0 1 2 0 0 0 0 0 2]
acc :  0.9444444444444444
"""