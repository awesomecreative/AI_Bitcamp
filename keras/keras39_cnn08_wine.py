import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten

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


x_train = x_train.reshape(142, 13, 1, 1)
x_test = x_test.reshape(36, 13, 1, 1)

#2. model
model = Sequential()
model.add(Conv2D(512, (2,1), input_shape=(13,1,1), padding='same', activation='relu'))
model.add(Conv2D(128, (2,1), activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='relu', input_shape=(13,)))
model.add(Dense(70, activation='sigmoid'))
model.add(Dense(120, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(80, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(3, activation='softmax'))


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
                     filepath = filepath + 'k39_08_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, verbose=1, callbacks=[es, mcp])

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
Epoch 87/1000
111/113 [============================>.] - ETA: 0s - loss: 8.9138e-08 - accuracy: 1.0000Restoring model weights from the end of the best epoch: 37.

Epoch 00087: val_loss did not improve from 0.00000
113/113 [==============================] - 1s 6ms/step - loss: 8.7561e-08 - accuracy: 1.0000 - val_loss: 0.0000e+00 - val_accuracy: 1.0000
Epoch 00087: early stopping
2/2 [==============================] - 0s 97ms/step - loss: 0.1623 - accuracy: 0.9722
loss :  0.16229934990406036
accuracy :  0.9722222089767456
[[1.54805446e-10 1.00000000e+00 3.02369307e-09]
 [2.44334109e-10 1.00000000e+00 4.21153512e-09]
 [9.99999285e-01 6.39607094e-07 6.01552159e-08]
 [1.89025702e-08 3.08902193e-09 1.00000000e+00]
 [9.99999881e-01 1.06884436e-07 2.01103614e-08]
 [1.66542460e-08 1.00000000e+00 3.00873104e-08]
 [1.00000000e+00 6.65591005e-09 3.46483664e-09]
 [1.23398669e-09 1.00000000e+00 2.64188458e-08]
 [1.00000000e+00 3.55490890e-08 2.28008066e-08]
 [1.97368760e-10 1.00000000e+00 3.13383341e-09]
 [3.48386386e-08 9.07908859e-09 1.00000000e+00]
 [1.54005877e-10 1.00000000e+00 2.89183388e-09]
 [8.44012915e-11 1.00000000e+00 1.77833770e-09]
 [1.00000000e+00 5.52390311e-09 6.75129241e-09]
 [5.46689138e-10 1.00000000e+00 7.68914354e-09]
 [1.13181214e-10 1.00000000e+00 2.26840235e-09]
 [2.89949259e-10 1.00000000e+00 4.21852198e-09]
 [2.54169823e-08 1.49238737e-08 1.00000000e+00]
 [2.03116670e-08 6.89965907e-09 1.00000000e+00]
 [3.16743289e-08 1.73822379e-08 1.00000000e+00]
 [5.72604075e-10 1.00000000e+00 7.26237737e-09]
 [4.39441727e-08 7.73924640e-08 9.99999881e-01]
 [1.00000000e+00 7.48118545e-09 6.30004227e-09]
 [1.35501921e-08 4.42924453e-09 1.00000000e+00]
 [1.07509072e-06 2.90109869e-03 9.97097850e-01]
 [1.97547109e-07 4.05269839e-06 9.99995708e-01]
 [2.88578800e-10 1.00000000e+00 4.64550132e-09]
 [1.00000000e+00 7.19520354e-09 1.09306031e-08]
 [1.11047144e-10 1.00000000e+00 2.28586239e-09]
 [1.52462274e-08 3.29917182e-09 1.00000000e+00]
 [1.00000000e+00 6.97834057e-09 5.95706329e-09]
 [9.99894381e-01 1.04112951e-04 1.58685475e-06]
 [1.00000000e+00 2.83359878e-08 1.00908544e-08]
 [9.99999404e-01 4.42930087e-07 6.51145555e-08]
 [1.00000000e+00 1.28207702e-08 1.75116863e-08]
 [5.02547834e-08 4.85307403e-08 9.99999881e-01]]
y_predict :  [1 1 0 2 0 1 0 1 0 1 2 1 1 0 1 1 1 2 2 2 1 2 0 2 2 2 1 0 1 2 0 0 0 0 0 2]
y_test :  [1 1 0 2 0 1 0 1 0 1 2 1 1 0 1 1 1 2 2 2 1 2 0 2 1 2 1 0 1 2 0 0 0 0 0 2]
acc :  0.9722222222222222
"""