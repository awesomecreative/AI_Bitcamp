from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, LSTM, Conv1D
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

x_train = x_train.reshape(120, 4, 1)
x_test = x_test.reshape(30, 4, 1)


#2. model
model = Sequential()
model.add(Conv1D(512, 1, input_shape=(4,1)))
model.add(Flatten())
model.add(Dense(50, activation = 'relu'))
model.add(Dense(40, activation = 'sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(10, activation = 'linear'))
model.add(Dense(3, activation = 'softmax'))


#3. compile, fit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', patience=1, restore_best_weights=True, verbose=1)

# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")

# filepath = './_save/MCP/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
#                      filepath = filepath + 'k39_07_' + date + '_' + filename)

model.fit(x_train, y_train, epochs=1, batch_size=1, validation_split=0.2, verbose=1,
          callbacks=[es])
# callbacks=[es,mcp]

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
"""
