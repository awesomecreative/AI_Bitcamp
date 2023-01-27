import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. data
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(1797, 64) (1797,)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[9])
# plt.show()

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
# scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (1437, 64) (360, 64)

x_train = x_train.reshape(1437,64,1)
x_test = x_test.reshape(360,64,1)

#2. model
model = Sequential()
model.add(Conv1D(512, 2, input_shape=(64,1)))
model.add(Flatten())
model.add(Dense(100, activation='linear'))
model.add(Dropout(0.5))
model.add(Dense(80, activation='relu'))
model.add(Dense(70, activation='sigmoid'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

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
#                      filepath = filepath + 'k39_09_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=1, callbacks=[es])
# callbacks=[es, mcp]

#4. evaluate, predict
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
y_test = np.argmax(y_test, axis=1)
print('y_predict : ', y_predict)
print('y_test___ : ', y_test)
acc = accuracy_score(y_test, y_predict)
print('accuracy : ', acc)

"""
Epoch 30/1000
36/36 [==============================] - ETA: 0s - loss: 0.0628 - accuracy: 0.9791Restoring model weights from the end of the best epoch: 20.
36/36 [==============================] - 0s 6ms/step - loss: 0.0628 - accuracy: 0.9791 - val_loss: 0.1407 - val_accuracy: 0.9514
Epoch 00030: early stopping
12/12 [==============================] - 0s 4ms/step - loss: 0.0687 - accuracy: 0.9833
loss :  0.06872253865003586
accuracy :  0.9833333492279053
y_predict :  [7 3 4 1 4 4 6 3 4 5 2 4 2 5 9 0 8 8 7 4 4 2 7 1 3 6 2 6 1 9 7 9 4 5 1 9 7
 2 4 7 3 7 6 7 4 8 5 1 2 1 2 7 6 1 1 2 8 8 6 5 6 2 3 4 0 7 9 3 5 0 4 4 1 8
 4 2 7 3 2 0 7 6 0 4 5 2 5 2 9 4 6 3 0 4 0 2 2 3 2 8 9 2 3 0 2 1 5 8 3 3 5
 5 1 6 5 6 9 7 8 2 0 9 8 4 5 8 0 8 5 7 9 5 8 9 4 1 9 6 4 1 9 2 5 5 6 0 5 2
 1 6 3 7 3 8 6 3 0 3 6 9 3 3 1 8 7 3 1 2 8 0 9 9 4 7 0 7 0 1 6 1 9 2 8 9 3
 6 0 1 8 8 0 7 0 9 0 4 9 9 2 1 2 5 5 9 8 0 3 5 0 6 1 4 1 9 8 6 8 4 5 8 6 1
 1 9 9 1 1 6 0 9 3 2 4 9 0 6 2 4 5 1 3 3 4 5 3 1 3 5 6 2 5 3 4 7 9 7 6 2 9
 0 6 0 8 4 0 4 7 7 0 6 0 3 1 0 6 8 2 3 2 3 3 6 2 9 8 0 6 5 7 1 2 6 7 1 5 2
 9 6 8 8 7 7 5 6 6 0 4 4 1 0 7 5 4 1 9 8 7 5 6 8 9 0 5 7 1 1 5 8 5 7 3 8 7
 3 5 8 2 3 7 3 8 7 3 1 9 0 8 9 0 7 4 7 0 5 1 1 9 4 5 3]
y_test___ :  [7 3 4 1 4 4 6 3 4 5 2 4 2 5 9 0 8 8 7 4 4 2 7 1 3 6 2 6 1 9 7 9 4 5 1 9 7
 2 4 7 3 7 6 7 4 8 5 1 2 1 2 7 6 1 1 2 8 8 6 8 6 2 3 4 0 7 9 3 5 0 4 4 1 8
 4 2 7 3 2 0 7 6 0 4 5 2 5 2 9 4 6 3 0 4 0 2 2 3 2 8 9 2 3 0 2 1 5 8 3 3 5
 5 1 6 5 6 1 9 8 2 0 9 8 4 5 8 0 8 5 7 9 5 8 9 4 1 9 6 4 1 9 2 5 5 6 0 5 2
 1 6 3 7 3 4 6 3 0 3 6 9 3 3 1 8 7 3 1 2 8 0 9 9 4 7 0 7 0 1 6 1 9 2 8 9 3
 6 0 6 8 8 0 7 0 9 0 4 9 9 2 1 2 5 5 9 8 0 3 5 0 6 1 4 1 9 8 6 8 4 5 8 6 1
 1 9 9 1 1 6 0 9 3 2 4 9 0 6 2 4 5 1 3 3 4 5 3 1 3 5 6 2 5 3 4 7 9 7 6 2 9
 0 6 0 8 4 0 4 7 7 0 6 0 3 1 0 6 8 2 3 2 3 3 6 2 9 8 0 6 5 7 1 2 6 7 1 5 2
 9 6 8 8 7 7 5 6 6 0 4 4 7 0 7 5 4 1 9 8 7 5 6 8 9 0 5 7 1 1 5 8 5 7 3 8 7
 3 5 8 2 3 7 3 8 7 3 1 9 0 8 9 0 7 4 7 0 5 1 1 9 4 5 3]
accuracy :  0.9833333333333333
"""