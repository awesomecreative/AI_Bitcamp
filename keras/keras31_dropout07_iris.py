from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout
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
print(y_train)
print(y_test)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# #2. model
# model = Sequential()
# model.add(Dense(50, activation = 'relu', input_shape=(4,)))
# model.add(Dense(40, activation = 'sigmoid'))
# model.add(Dropout(0.3))
# model.add(Dense(30, activation = 'relu'))
# model.add(Dense(10, activation = 'linear'))
# model.add(Dense(3, activation = 'softmax'))

#2. model
input1 = Input(shape=(4,))
dense1 = Dense(50, activation='relu')(input1)
dense2 = Dense(40, activation='sigmoid')(dense1)
drop2 = Dropout(0,3)(dense2)
dense3 = Dense(30, activation='relu')(drop2)
dense4 = Dense(10, activation='linear')(dense3)
output1 = Dense(3, activation='softmax')(dense4)
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
                     filepath = filepath + 'k31_07_' + date + '_' + filename)

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
