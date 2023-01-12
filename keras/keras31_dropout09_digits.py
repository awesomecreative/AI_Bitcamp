import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
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

# #2. model
# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(64,)))
# model.add(Dense(50, activation='sigmoid'))
# model.add(Dense(100, activation='linear'))
# model.add(Dropout(0.5))
# model.add(Dense(80, activation='relu'))
# model.add(Dense(70, activation='sigmoid'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(10, activation='softmax'))

#2. model
input1 = Input(shape=(64,))
dense1 = Dense(10, activation='relu')(input1)
dense2 = Dense(50, activation='sigmoid')(dense1)
dense3 = Dense(100, activation='linear')(dense2)
drop3 = Dropout(0.5)(dense3)
dense4 = Dense(80, activation='relu')(drop3)
dense5 = Dense(70, activation='sigmoid')(dense4)
dense6 = Dense(30, activation='relu')(dense5)
output1 = Dense(10, activation='softmax')(dense6)
model = Model(inputs=input1, outputs=output1)

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
                     filepath = filepath + 'k31_09_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, verbose=1, callbacks=[es, mcp])

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
