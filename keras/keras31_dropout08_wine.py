import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout

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

# #2. model
# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(13,)))
# model.add(Dense(70, activation='sigmoid'))
# model.add(Dense(120, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(80, activation='linear'))
# model.add(Dense(10, activation='linear'))
# model.add(Dense(3, activation='softmax'))

#2. model
input1 = Input(shape=(13,))
dense1 = Dense(10, activation='relu')(input1)
dense2 = Dense(70, activation='sigmoid')(dense1)
dense3 = Dense(120, activation='relu')(dense2)
drop3 = Dropout(0.5)(dense3)
dense4 = Dense(80, activation='linear')(drop3)
dense5 = Dense(10, activation='linear')(dense4)
output1 = Dense(3, activation='softmax')(dense5)
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
                     filepath = filepath + 'k31_08_' + date + '_' + filename)

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
