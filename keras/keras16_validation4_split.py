import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. data
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(
   x, y, test_size=0.2, random_state=123)

print(x_train.shape, x_test.shape) # (12,) (4, )
print(y_train.shape, y_test.shape) # (12,) (4, )

#2. model
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

#3. compile, fit(train, validation)
model.compile(loss='mse', optimizer='adam', metrics=['mae','mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.25)

#4. evaluate, predict 
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

result = model.predict([17])
print('result of 17 : ', result)
