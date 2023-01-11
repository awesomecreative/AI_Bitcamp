from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
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

# #2. model
# model = Sequential()
# model.add(Dense(50, input_shape=(30,), activation='linear'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

#2. model
input1 = Input(shape=(30,))
dense1 = Dense(50, activation='linear')(input1)
dense2 = Dense(40, activation='relu')(dense1)
dense3 = Dense(30, activation='relu')(dense2)
dense4 = Dense(10, activation='relu')(dense3)
output1 = Dense(1, activation='sigmoid')(dense4)
model = Model(inputs=input1, outputs=output1)

#3. compile, fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True, verbose=1)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100, validation_split=0.2, verbose=1,
                 callbacks=[earlyStopping])

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
