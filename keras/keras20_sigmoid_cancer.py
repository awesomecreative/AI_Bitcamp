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

hist = model.fit(x_train, y_train, epochs=1, batch_size=100, validation_split=0.2, verbose=1,
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

#5. draw, submit


"""
Epoch 24/300
357/364 [============================>.] - ETA: 0s - loss: 0.2249 - accuracy: 0.9160Restoring model weights from the end of the best epoch: 14.
364/364 [==============================] - 0s 704us/step - loss: 0.2287 - accuracy: 0.9121 - val_loss: 0.2227 - val_accuracy: 0.9341
Epoch 00024: early stopping
4/4 [==============================] - 0s 996us/step - loss: 0.2457 - accuracy: 0.9123
loss :  0.24569936096668243
accuracy :  0.9122806787490845

Epoch 16/300
359/364 [============================>.] - ETA: 0s - loss: 0.6560 - accuracy: 0.6351Restoring model weights from the end of the best epoch: 6.
364/364 [==============================] - 0s 694us/step - loss: 0.6545 - accuracy: 0.6374 - val_loss: 0.6794 - val_accuracy: 0.6044
Epoch 00016: early stopping
4/4 [==============================] - 0s 0s/step - loss: 0.2093 - accuracy: 0.9386
loss :  0.20929409563541412
accuracy :  0.9385964870452881
"""

"""
Memo
이진분류 : 마지막 y 값이 0, 1 중 하나만 나와야한다.
따라서 마지막 layer의 activation='sigmoid' 여야 한다.
그리고 loss='binary_crossentropy'이다. 또한, metrics=['accuracy'] 쓰기.

ValueError: Classification metrics can't handle a mix of binary and continuous targets
y_predict는 실수로 출력되어 있고 y_test에는 1,0으로만 되어있음.
따라서 자료형이 맞지 않다고 오류 뜨는 것임.

해결하는 방법은?
y_predict = y_predict.astype(int)
astype이란 파라미터를 써서 자료형을 integer(정수형)으로 바꿔준다.
astype은 numpy란 라이브러리에 있기 때문에 import numpy 해줘야 한다.

# sigmoid는 0에서 1사이를 출력한다. 0과 1만을 출력하는 것이 아니다.
"""