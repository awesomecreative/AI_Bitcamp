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

# import matplotlib.pyplot as plt
# plt.gray()
# plt.matshow(datasets.images[9])
# plt.show()

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1, stratify=y)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

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
1) scaling 안 했을 때
Epoch 65/1000
1118/1149 [============================>.] - ETA: 0s - loss: 0.0249 - accuracy: 0.9902Restoring model weights from the end of the best epoch: 15.
1149/1149 [==============================] - 1s 912us/step - loss: 0.0243 - accuracy: 0.9904 - val_loss: 0.2358 - val_accuracy: 0.9479
Epoch 00065: early stopping
12/12 [==============================] - 0s 1ms/step - loss: 0.1453 - accuracy: 0.9667
loss :  0.1453431248664856
accuracy :  0.9666666388511658
y_predict :  [7 3 4 1 4 4 6 3 4 5 2 4 2 5 9 0 1 8 7 4 4 8 7 1 3 6 2 6 1 9 7 9 4 5 1 9 7
 2 4 7 3 7 6 7 4 8 5 1 2 1 2 7 6 1 1 2 8 8 6 8 6 2 3 4 0 7 3 3 5 0 4 4 1 8
 4 2 7 3 2 0 7 6 0 4 5 2 5 2 9 4 6 3 0 4 0 2 2 3 2 8 9 2 3 0 2 1 5 8 3 3 5
 5 1 6 5 6 1 7 9 2 0 9 8 4 5 8 0 8 5 7 9 5 8 9 4 1 9 6 4 1 9 2 5 5 6 0 5 2
 1 6 3 7 3 8 6 3 0 3 6 3 3 3 1 8 7 3 1 2 8 0 9 9 4 7 0 7 0 1 6 1 9 2 8 9 3
 6 0 1 8 8 0 7 0 9 0 4 9 9 2 1 3 5 5 9 8 0 3 5 0 6 1 4 1 9 8 6 8 4 5 8 6 1
 1 9 9 1 1 6 0 9 3 2 4 9 0 6 2 7 5 1 3 3 7 5 3 1 3 5 6 2 5 3 4 7 9 7 6 2 9
 0 6 0 8 4 0 4 7 7 0 6 0 3 1 0 6 8 2 3 2 3 3 6 2 9 8 0 6 5 7 1 2 6 7 1 5 2
 9 6 8 8 7 7 5 6 6 0 4 4 7 0 7 5 4 1 9 8 7 5 6 8 9 0 5 7 1 1 5 8 5 7 3 8 7
 3 5 8 2 3 7 3 8 7 3 1 3 0 8 9 0 7 4 7 0 5 1 1 9 4 5 3]
y_test___ :  [7 3 4 1 4 4 6 3 4 5 2 4 2 5 9 0 8 8 7 4 4 2 7 1 3 6 2 6 1 9 7 9 4 5 1 9 7
 2 4 7 3 7 6 7 4 8 5 1 2 1 2 7 6 1 1 2 8 8 6 8 6 2 3 4 0 7 9 3 5 0 4 4 1 8
 4 2 7 3 2 0 7 6 0 4 5 2 5 2 9 4 6 3 0 4 0 2 2 3 2 8 9 2 3 0 2 1 5 8 3 3 5
 5 1 6 5 6 1 9 8 2 0 9 8 4 5 8 0 8 5 7 9 5 8 9 4 1 9 6 4 1 9 2 5 5 6 0 5 2
 1 6 3 7 3 4 6 3 0 3 6 9 3 3 1 8 7 3 1 2 8 0 9 9 4 7 0 7 0 1 6 1 9 2 8 9 3
 6 0 6 8 8 0 7 0 9 0 4 9 9 2 1 2 5 5 9 8 0 3 5 0 6 1 4 1 9 8 6 8 4 5 8 6 1
 1 9 9 1 1 6 0 9 3 2 4 9 0 6 2 4 5 1 3 3 4 5 3 1 3 5 6 2 5 3 4 7 9 7 6 2 9
 0 6 0 8 4 0 4 7 7 0 6 0 3 1 0 6 8 2 3 2 3 3 6 2 9 8 0 6 5 7 1 2 6 7 1 5 2
 9 6 8 8 7 7 5 6 6 0 4 4 7 0 7 5 4 1 9 8 7 5 6 8 9 0 5 7 1 1 5 8 5 7 3 8 7
 3 5 8 2 3 7 3 8 7 3 1 9 0 8 9 0 7 4 7 0 5 1 1 9 4 5 3]
accuracy :  0.9666666666666667

2) MinMax
Epoch 159/1000
1096/1149 [===========================>..] - ETA: 0s - loss: 1.1772e-06 - accuracy: 1.0000Restoring model weights from the end of the best epoch: 109.
1149/1149 [==============================] - 1s 864us/step - loss: 1.1284e-06 - accuracy: 1.0000 - val_loss: 0.7362 - val_accuracy: 0.9444
Epoch 00159: early stopping
12/12 [==============================] - 0s 1000us/step - loss: 0.2579 - accuracy: 0.9667
loss :  0.2578933835029602
accuracy :  0.9666666388511658
y_predict :  [7 3 4 2 4 4 6 3 4 5 2 4 2 5 9 0 8 8 7 4 4 2 7 1 3 6 2 6 1 9 7 8 4 5 1 9 7
 2 4 7 3 7 6 7 4 8 5 1 2 1 2 7 6 1 1 2 8 8 6 3 6 2 3 4 0 7 9 3 5 0 4 4 1 8
 4 2 7 3 2 0 7 6 0 4 5 2 5 2 9 4 6 3 0 4 0 2 2 3 2 8 9 2 3 0 2 1 5 8 3 3 5
 5 1 6 5 6 9 7 9 2 0 9 8 4 5 8 0 8 5 7 9 5 8 9 4 1 9 6 4 1 9 2 5 5 6 0 5 2
 1 6 3 7 3 4 6 3 0 3 6 9 3 3 1 8 7 3 1 2 8 0 9 9 4 7 0 7 0 1 6 1 9 2 8 9 3
 6 0 1 8 8 0 7 0 9 0 4 9 9 2 1 2 5 5 9 8 0 3 5 0 6 1 4 1 9 8 6 8 4 5 8 6 1
 1 9 9 8 1 6 0 9 3 2 4 9 0 6 2 7 5 1 3 3 4 5 3 1 3 5 6 2 5 3 4 7 9 7 6 2 8
 0 6 0 8 4 0 4 7 7 0 6 0 3 1 0 6 8 2 3 2 3 3 6 2 9 8 0 6 5 7 1 2 6 7 1 5 2
 9 6 8 9 7 7 5 6 6 0 4 4 5 0 7 5 4 1 9 8 7 5 6 8 9 0 5 7 1 1 5 8 5 7 3 8 7
 3 5 8 2 3 7 3 8 7 3 1 9 0 8 9 0 7 4 7 0 5 1 1 9 4 5 3]
y_test___ :  [7 3 4 1 4 4 6 3 4 5 2 4 2 5 9 0 8 8 7 4 4 2 7 1 3 6 2 6 1 9 7 9 4 5 1 9 7
 2 4 7 3 7 6 7 4 8 5 1 2 1 2 7 6 1 1 2 8 8 6 8 6 2 3 4 0 7 9 3 5 0 4 4 1 8
 4 2 7 3 2 0 7 6 0 4 5 2 5 2 9 4 6 3 0 4 0 2 2 3 2 8 9 2 3 0 2 1 5 8 3 3 5
 5 1 6 5 6 1 9 8 2 0 9 8 4 5 8 0 8 5 7 9 5 8 9 4 1 9 6 4 1 9 2 5 5 6 0 5 2
 1 6 3 7 3 4 6 3 0 3 6 9 3 3 1 8 7 3 1 2 8 0 9 9 4 7 0 7 0 1 6 1 9 2 8 9 3
 6 0 6 8 8 0 7 0 9 0 4 9 9 2 1 2 5 5 9 8 0 3 5 0 6 1 4 1 9 8 6 8 4 5 8 6 1
 1 9 9 1 1 6 0 9 3 2 4 9 0 6 2 4 5 1 3 3 4 5 3 1 3 5 6 2 5 3 4 7 9 7 6 2 9
 0 6 0 8 4 0 4 7 7 0 6 0 3 1 0 6 8 2 3 2 3 3 6 2 9 8 0 6 5 7 1 2 6 7 1 5 2
 9 6 8 8 7 7 5 6 6 0 4 4 7 0 7 5 4 1 9 8 7 5 6 8 9 0 5 7 1 1 5 8 5 7 3 8 7
 3 5 8 2 3 7 3 8 7 3 1 9 0 8 9 0 7 4 7 0 5 1 1 9 4 5 3]
accuracy :  0.9666666666666667

3) Standard
Epoch 68/1000
1125/1149 [============================>.] - ETA: 0s - loss: 0.0755 - accuracy: 0.9813Restoring model weights from the end of the best epoch: 18.
1149/1149 [==============================] - 1s 848us/step - loss: 0.0740 - accuracy: 0.9817 - val_loss: 0.2415 - val_accuracy: 0.9549
Epoch 00068: early stopping
12/12 [==============================] - 0s 1ms/step - loss: 0.3229 - accuracy: 0.9361
loss :  0.3229321837425232
accuracy :  0.9361110925674438
y_predict :  [7 8 4 1 4 4 6 3 4 5 2 4 2 5 3 0 1 8 7 4 4 8 7 1 3 6 8 6 1 9 7 9 4 5 1 8 7
 2 4 7 3 7 6 7 4 8 5 1 2 1 2 7 6 1 1 2 8 8 6 3 6 2 3 4 0 7 9 3 5 0 4 4 1 8
 4 2 7 3 2 0 7 6 0 4 9 2 5 2 9 4 6 3 0 4 0 2 2 3 2 8 9 2 3 0 2 1 5 8 3 3 5
 5 1 6 5 6 1 9 8 2 0 9 8 4 5 8 0 8 5 7 9 5 8 9 4 1 9 6 4 8 9 2 5 2 6 0 5 2
 1 6 5 7 3 9 6 3 0 3 6 9 3 3 1 8 7 3 1 2 8 0 9 9 4 7 0 7 0 1 6 1 9 2 8 9 3
 6 0 1 8 8 0 7 0 9 0 4 9 9 2 1 2 5 5 9 8 0 3 5 0 6 1 4 1 9 8 6 8 4 5 8 6 1
 1 9 9 1 1 6 0 9 3 2 4 9 0 6 2 9 5 1 3 3 1 5 3 1 3 5 6 2 5 3 4 7 9 4 6 2 9
 0 6 0 1 4 0 4 7 7 0 6 0 3 1 0 6 8 2 3 2 3 3 6 2 9 8 0 6 5 7 1 2 6 9 1 5 2
 9 6 8 8 7 7 5 6 6 0 4 4 9 0 7 5 4 1 9 8 7 5 6 8 9 0 5 7 1 1 5 8 5 7 3 8 7
 9 5 9 2 3 7 3 8 7 3 1 8 0 8 9 0 7 4 9 0 5 1 1 9 4 5 3]
y_test___ :  [7 3 4 1 4 4 6 3 4 5 2 4 2 5 9 0 8 8 7 4 4 2 7 1 3 6 2 6 1 9 7 9 4 5 1 9 7
 2 4 7 3 7 6 7 4 8 5 1 2 1 2 7 6 1 1 2 8 8 6 8 6 2 3 4 0 7 9 3 5 0 4 4 1 8
 4 2 7 3 2 0 7 6 0 4 5 2 5 2 9 4 6 3 0 4 0 2 2 3 2 8 9 2 3 0 2 1 5 8 3 3 5
 5 1 6 5 6 1 9 8 2 0 9 8 4 5 8 0 8 5 7 9 5 8 9 4 1 9 6 4 1 9 2 5 5 6 0 5 2
 1 6 3 7 3 4 6 3 0 3 6 9 3 3 1 8 7 3 1 2 8 0 9 9 4 7 0 7 0 1 6 1 9 2 8 9 3
 6 0 6 8 8 0 7 0 9 0 4 9 9 2 1 2 5 5 9 8 0 3 5 0 6 1 4 1 9 8 6 8 4 5 8 6 1
 1 9 9 1 1 6 0 9 3 2 4 9 0 6 2 4 5 1 3 3 4 5 3 1 3 5 6 2 5 3 4 7 9 7 6 2 9
 0 6 0 8 4 0 4 7 7 0 6 0 3 1 0 6 8 2 3 2 3 3 6 2 9 8 0 6 5 7 1 2 6 7 1 5 2
 9 6 8 8 7 7 5 6 6 0 4 4 7 0 7 5 4 1 9 8 7 5 6 8 9 0 5 7 1 1 5 8 5 7 3 8 7
 3 5 8 2 3 7 3 8 7 3 1 9 0 8 9 0 7 4 7 0 5 1 1 9 4 5 3]
accuracy :  0.9361111111111111

: Digits 에서는 scaling전이랑 MinMax가 비슷하게 좋게 나오고 Standard는 별로 안 좋게 나왔다.
"""