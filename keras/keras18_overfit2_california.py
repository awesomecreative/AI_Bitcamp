import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. data
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=123)

#2. model
model = Sequential()
model.add(Dense(16, input_dim=8, activation = 'linear'))
model.add(Dense(64, activation = 'sigmoid'))
model.add(Dense(128, activation = 'sigmoid'))
model.add(Dense(512, activation = 'sigmoid'))
model.add(Dense(64, activation = 'sigmoid'))
model.add(Dense(16, activation = 'sigmoid'))
model.add(Dense(1, activation = 'linear'))

#3. compile, fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
hist = model.fit(x_train,y_train, epochs=500, batch_size=50, validation_split=0.2)

#4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

print("=======================================")
print(hist) # <keras.callbacks.History object at 0x00000195CE646850>
print("=======================================")
print(hist.history)
print("=======================================")
print(hist.history['loss'])
print("=======================================")
print(hist.history['val_loss'])
print("=======================================")

#5. draw, submit
import matplotlib.pyplot as plt

plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('california loss')
plt.legend(loc='center right')
plt.show()