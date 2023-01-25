from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten
from sklearn.model_selection import train_test_split

#1. data
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets['target']
print(x)
print(y)
print(x.shape, y.shape) # (150, 4) (150,) => input_dim=4고 마지막 Dense는 1이겠군!

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)
print(y.shape) #(150,3)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape) # (120, 4) (30, 4)


x_train = x_train.reshape(120, 2, 2, 1)
x_test = x_test.reshape(30, 2, 2, 1)


#2. model
model = Sequential()
model.add(Conv2D(512, (2,2), input_shape=(2,2,1), padding='same', activation='relu'))
model.add(Conv2D(128, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(50, activation = 'relu', input_shape=(4,)))
model.add(Dense(40, activation = 'sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(10, activation = 'linear'))
model.add(Dense(3, activation = 'softmax'))


#3. compile, fit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True, verbose=1)

import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

filepath = './_save/MCP/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                     filepath = filepath + 'k39_07_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=1,
          callbacks=[es, mcp])

#4. evaluate, predict
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1)
print('y_pred : ', y_predict)
print('y_test : ', y_test)
y_test = np.argmax(y_test, axis=1)
print('y_test : ', y_test)
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

"""
Epoch 68/100
94/96 [============================>.] - ETA: 0s - loss: 0.0017 - accuracy: 1.0000Restoring model weights from the end of the best epoch: 18.

Epoch 00068: val_loss did not improve from 0.10474
96/96 [==============================] - 1s 6ms/step - loss: 0.0016 - accuracy: 1.0000 - val_loss: 0.7432 - val_accuracy: 0.9167
Epoch 00068: early stopping
1/1 [==============================] - 0s 222ms/step - loss: 0.0185 - accuracy: 1.0000
loss :  0.01845855452120304
accuracy :  1.0
y_pred :  [0 2 1 2 2 2 0 0 0 1 2 1 0 1 0 2 0 1 1 0 1 1 2 2 0 0 2 2 1 1]
y_test :  [[1. 0. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 0. 1.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [1. 0. 0.]
 [1. 0. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]]
y_test :  [0 2 1 2 2 2 0 0 0 1 2 1 0 1 0 2 0 1 1 0 1 1 2 2 0 0 2 2 1 1]
acc :  1.0
"""
