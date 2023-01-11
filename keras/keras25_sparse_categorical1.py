# <sparse_categorical_crossentropy> : load_iris

from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y)
# print(y.shape) #(150,3)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1, stratify=y)
print(y_train)
print(y_test)

#2. model
model = Sequential()
model.add(Dense(50, activation = 'relu', input_shape=(4,)))
model.add(Dense(40, activation = 'sigmoid'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(10, activation = 'linear'))
model.add(Dense(3, activation = 'softmax'))

#3. compile, fit
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=1)

#4. evaluate, predict
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

# print(y_test[:5])
# y_predict = model.predict(x_test[:5])
# print(y_predict)

from sklearn.metrics import accuracy_score
import numpy as np
y_predict = model.predict(x_test) # argmax하기 전, 단순히 predict한 경우에는 실수값이 있는 벡터 데이터가 나옴. 
print(y_predict.shape)
y_predict = np.argmax(y_predict, axis=1) # argmax 후의 y_predict는 0,1,2 정수값만 나옴.
print('argmax 후 y_pred : ', y_predict)
# y_test = np.argmax(y_test, axis=1) # 원핫을 안했으므로 여기서 argmax 쓸 필요가 없음.
print('y_test : ', y_test)
acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)

"""
Epoch 100/100
96/96 [==============================] - 0s 814us/step - loss: 0.0965 - accuracy: 0.9792 - val_loss: 0.0379 - val_accuracy: 1.0000
1/1 [==============================] - 0s 78ms/step - loss: 0.0151 - accuracy: 1.0000
loss :  0.015066159889101982
accuracy :  1.0
(30, 3)
argmax 후 y_pred :  [2 0 1 0 0 0 2 2 2 1 0 1 2 1 2 0 2 1 1 2 1 1 0 0 2 2 0 0 1 1]
y_test :  [2 0 1 0 0 0 2 2 2 1 0 1 2 1 2 0 2 1 1 2 1 1 0 0 2 2 0 0 1 1]
acc :  1.0
"""

"""
<sparse_categorical_crossentropy>
sparse_categorical_crossentropy로 다중 분류할 때
softmax의 node의 개수는 one-hot 하지 않았지만 one-hot한 것처럼 y의 class 개수를 적어준다.
"""
