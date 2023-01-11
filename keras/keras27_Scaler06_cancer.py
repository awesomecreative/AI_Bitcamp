from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. model

model = Sequential()
model.add(Dense(50, input_shape=(30,), activation='linear'))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

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


"""
1) scaling 안 했을 때
Epoch 84/1000
1/4 [======>.......................] - ETA: 0s - loss: 0.1853 - accuracy: 0.9600Restoring model weights from the end of the best epoch: 74.
4/4 [==============================] - 0s 15ms/step - loss: 0.3443 - accuracy: 0.9148 - val_loss: 0.3158 - val_accuracy: 0.8681
Epoch 00084: early stopping
4/4 [==============================] - 0s 3ms/step - loss: 0.1894 - accuracy: 0.9474
loss :  0.1894075721502304
accuracy :  0.9473684430122375
accuracy_score :  0.3684210526315789

2) MinMaxScaler
Epoch 65/1000
1/4 [======>.......................] - ETA: 0s - loss: 0.0350 - accuracy: 0.9900Restoring model weights from the end of the best epoch: 55.
4/4 [==============================] - 0s 11ms/step - loss: 0.0393 - accuracy: 0.9890 - val_loss: 0.1505 - val_accuracy: 0.9451
Epoch 00065: early stopping
4/4 [==============================] - 0s 2ms/step - loss: 0.0706 - accuracy: 0.9825
loss :  0.07061837613582611
accuracy :  0.9824561476707458
accuracy_score :  0.3684210526315789

3) StandardScaler
Epoch 23/1000
1/4 [======>.......................] - ETA: 0s - loss: 0.0385 - accuracy: 0.9900Restoring model weights from the end of the best epoch: 13.
4/4 [==============================] - 0s 12ms/step - loss: 0.0367 - accuracy: 0.9863 - val_loss: 0.1038 - val_accuracy: 0.9780
Epoch 00023: early stopping
4/4 [==============================] - 0s 3ms/step - loss: 0.1191 - accuracy: 0.9649
loss :  0.1191210001707077
accuracy :  0.9649122953414917
accuracy_score :  0.3684210526315789

: Cancer에서는 MinMaxScaler가 가장 좋게 나온다.
"""