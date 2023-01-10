import numpy as np
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. data
datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) #(1797, 64) (1797,)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

import matplotlib.pyplot as plt
plt.gray()
plt.matshow(datasets.images[9])
plt.show()

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1)

#2. model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(64,)))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(100, activation='linear'))
model.add(Dense(80, activation='relu'))
model.add(Dense(70, activation='sigmoid'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. compile, fit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True, verbose=1)
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, verbose=1, callbacks=[earlyStopping])

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
Epoch 76/100
1081/1149 [===========================>..] - ETA: 0s - loss: 0.0664 - accuracy: 0.9796Restoring model weights from the end of the best epoch: 26.
1149/1149 [==============================] - 1s 860us/step - loss: 0.0811 - accuracy: 0.9774 - val_loss: 0.5242 - val_accuracy: 0.9410
Epoch 00076: early stopping
12/12 [==============================] - 0s 957us/step - loss: 0.2141 - accuracy: 0.9472
loss :  0.21411268413066864
accuracy :  0.9472222328186035
y_predict :  [1 5 0 7 1 0 6 1 5 4 9 2 1 8 4 6 9 3 7 4 7 1 8 6 0 9 6 1 3 7 5 9 8 3 2 8 8
 1 1 0 7 9 0 0 8 7 2 7 4 3 4 3 4 0 4 7 0 5 9 5 2 1 7 0 5 1 8 3 3 4 0 3 7 4
 3 4 2 9 7 3 2 5 3 4 1 5 5 2 8 2 2 2 2 7 0 8 8 7 4 2 3 8 2 3 3 0 2 9 3 2 3
 2 3 1 1 9 1 2 0 4 8 5 4 4 7 6 8 6 6 1 7 5 6 3 8 3 7 1 8 8 3 4 7 8 5 0 6 0
 6 3 7 6 5 6 2 2 2 3 0 7 6 5 6 4 1 0 6 0 6 4 0 9 3 8 1 2 3 1 9 0 7 6 2 9 3
 8 3 4 6 3 3 7 4 8 2 7 6 1 6 8 4 0 3 8 0 9 9 9 4 1 3 6 8 0 9 5 9 8 2 3 5 3
 0 8 7 4 0 3 3 3 6 3 3 2 9 1 6 9 0 4 2 2 7 9 1 6 7 6 3 5 1 9 3 4 0 6 4 8 3
 3 6 3 1 4 0 4 4 8 7 9 1 5 2 7 0 8 0 4 4 0 1 0 6 4 2 8 5 0 2 6 0 1 8 2 0 9
 5 6 2 0 5 0 9 1 4 7 1 7 0 6 6 8 0 2 2 6 9 9 7 5 1 7 6 4 6 1 9 4 7 1 3 7 8
 8 6 9 8 3 2 4 8 7 5 8 6 9 9 8 5 0 0 4 9 8 0 4 9 4 2 5]
y_test___ :  [1 5 0 7 1 0 6 1 5 4 9 2 7 8 4 6 9 3 7 4 7 1 8 6 0 9 6 1 3 7 5 9 8 3 2 8 8
 1 1 0 7 9 0 0 8 7 2 7 4 3 4 3 4 0 4 7 0 5 5 5 2 1 7 0 5 1 8 3 3 4 0 3 7 4
 3 4 2 9 7 3 2 5 3 4 1 5 5 2 5 2 2 2 2 7 0 8 1 7 4 2 3 8 2 3 3 0 2 9 9 2 3
 2 8 1 1 9 1 2 0 4 8 5 4 4 7 6 7 6 6 1 7 5 6 3 8 3 7 1 8 5 3 4 7 8 5 0 6 0
 6 3 7 6 5 6 2 2 2 3 0 7 6 5 6 4 1 0 6 0 6 4 0 9 3 8 1 2 3 1 9 0 7 6 2 9 3
 5 3 4 6 3 3 7 4 9 2 7 6 1 6 8 4 0 3 1 0 9 9 9 0 1 8 6 8 0 9 5 9 8 2 3 5 3
 0 8 7 4 0 3 3 3 6 3 3 2 9 1 6 9 0 4 2 2 7 9 1 6 7 6 3 7 1 9 3 4 0 6 4 8 5
 3 6 3 1 4 0 4 4 8 7 9 1 5 2 7 0 9 0 4 4 0 1 0 6 4 2 8 5 0 2 6 0 1 8 2 0 9
 5 6 2 0 5 0 9 1 4 7 1 7 0 6 6 8 0 2 2 6 9 9 7 5 1 7 6 4 6 1 9 4 7 1 3 7 8
 1 6 9 8 3 2 4 8 7 5 5 6 9 9 8 5 0 0 4 9 3 0 4 9 4 2 5]
accuracy :  0.9472222222222222
"""

"""
Memo
이미지 연산 또한 행렬로 표현이 가능하다.
흑백은 표1개, 컬러는 표3개 (흑백 뒤에 2장 더 들어감)
즉, 이미지는 다중분류 데이터이다.
"""