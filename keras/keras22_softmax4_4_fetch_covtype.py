# 판다스 겟더미 _ tf.argmax 사용하는 것 : 결과값 데이터 형태만 다를뿐 결과값 나옴!
# from pandas as pd / y = pd.get_dummies(y)

import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. data
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)
print(np.unique(y, return_counts=True))
# (581012, 54) (581012,)
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))
# => shape와 np.unique를 확인한 결과 이 데이터는 회귀가 아닌 다중분류이다.

import pandas as pd
y = pd.get_dummies(y)
print(y)
print(y.shape) # (581012, 7)
print(type(y)) # <class 'pandas.core.frame.DataFrame'>

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=1)

#2. model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(54,)))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(120, activation='linear'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(10, activation='relu'))
model.add(Dense(7, activation='softmax'))

#3. compile, fit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True, verbose=1)
import time
start = time.time()
model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.2, verbose=1, callbacks=[earlyStopping])
end = time.time()

#4. evaluate, predict
loss, accuracy = model.evaluate(x_test, y_test)
print('loss : ', loss)
print('accuracy : ', accuracy)

y_predict = model.predict(x_test)
print('y_pred_1 : ', y_predict)
print('y_test_1 : ', y_test)
print(type(y_predict))
print(type(y_test))

"""
import pandas as pd
y_test = y_test.values
# y_test = y_test.to_numpy() 로 해줘도 된다.
print(y_test)
print(type(y_test))
"""

import tensorflow as tf

y_predict = tf.argmax(y_predict, axis=1)
print('y_pred_2 : ', y_predict)
print(type(y_predict))

y_test = tf.argmax(y_test, axis=1)
print('y_test_2 : ', y_test)
print(type(y_test))

acc = accuracy_score(y_test, y_predict)
print('acc : ', acc)
print('time : ', end-start)

"""
Epoch 40/100
3714/3719 [============================>.] - ETA: 0s - loss: 0.6399 - accuracy: 0.7218Restoring model weights from the end of the best epoch: 35.
3719/3719 [==============================] - 21s 6ms/step - loss: 0.6399 - accuracy: 0.7218 - val_loss: 0.6431 - val_accuracy: 0.7144      
Epoch 00040: early stopping
3632/3632 [==============================] - 12s 3ms/step - loss: 0.6142 - accuracy: 0.7380
loss :  0.6141788363456726
accuracy :  0.7380102276802063
y_pred_1 :  [[1.3032533e-02 9.7872353e-01 3.1726067e-03 ... 2.5175232e-03
  2.4945680e-03 5.9198741e-05]
 [7.6201659e-01 2.2907570e-01 4.2407120e-08 ... 1.2108919e-04
  2.3335422e-05 8.7632900e-03]
 [4.7332380e-04 1.6876187e-02 6.5689003e-01 ... 1.9425624e-04
  2.8516623e-01 5.9818467e-06]
 ...
 [7.5592011e-01 2.4267593e-01 1.0115461e-06 ... 1.0722914e-03
  1.3122341e-06 3.2938950e-04]
 [6.2589359e-01 3.7292093e-01 5.8968806e-07 ... 2.1570470e-04
  2.7846834e-06 9.6633518e-04]
 [7.8385729e-01 2.1519935e-01 2.0412382e-07 ... 5.3812948e-04
  5.5026834e-07 4.0442182e-04]]
y_test_1 :          1  2  3  4  5  6  7
376969  0  1  0  0  0  0  0
59897   1  0  0  0  0  0  0
247100  0  0  0  0  0  1  0
111532  0  1  0  0  0  0  0
522294  1  0  0  0  0  0  0
...    .. .. .. .. .. .. ..
72376   0  1  0  0  0  0  0
93646   0  1  0  0  0  0  0
180759  1  0  0  0  0  0  0
561349  1  0  0  0  0  0  0
209740  1  0  0  0  0  0  0

[116203 rows x 7 columns]
<class 'numpy.ndarray'>
<class 'pandas.core.frame.DataFrame'>
y_pred_2 :  tf.Tensor([1 0 2 ... 0 0 0], shape=(116203,), dtype=int64)
<class 'tensorflow.python.framework.ops.EagerTensor'>
y_test_2 :  tf.Tensor([1 0 5 ... 0 0 0], shape=(116203,), dtype=int64)
<class 'tensorflow.python.framework.ops.EagerTensor'>
acc :  0.7380102062769464
time :  857.9710402488708
"""

"""
<scikit-onehotencoder vs pandas-get_dummies vs keras-to_categorical>
get_dummies를 쓰면 자료형이 <class 'pandas.core.frame.DataFrame'>이다.
여기서 굳이 자료형을 <class 'numpy.ndarray'>로 바꾸지 않고
np.argmax를 tf.argmax로 바꿔서 결과를 구할수도 있다.
대신 마지막 결과에 나오는 데이터형이 <class 'numpy.ndarray'>가 아니라
<class 'tensorflow.python.framework.ops.EagerTensor'> 이다.
"""