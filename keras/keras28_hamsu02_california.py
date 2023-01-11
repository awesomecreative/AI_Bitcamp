import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#1. data
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=123)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# #2. model
# model = Sequential()
# model.add(Dense(16, input_dim=8, activation = 'linear'))
# model.add(Dense(64, activation = 'sigmoid'))
# model.add(Dense(128, activation = 'sigmoid'))
# model.add(Dense(512, activation = 'sigmoid'))
# model.add(Dense(64, activation = 'sigmoid'))
# model.add(Dense(16, activation = 'sigmoid'))
# model.add(Dense(1, activation = 'linear'))

#2. model
input1 = Input(shape=(8,))
dense1 = Dense(16, activation='linear')(input1)
dense2 = Dense(64, activation='sigmoid')(dense1)
dense3 = Dense(128, activation='sigmoid')(dense2)
dense4 = Dense(512, activation='sigmoid')(dense3)
dense5 = Dense(64, activation='sigmoid')(dense4)
dense6 = Dense(16, activation='sigmoid')(dense5)
output1 = Dense(1, activation='linear')(dense6)
model = Model(inputs=input1, outputs=output1)
model.summary()

#3. compile, fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)
hist = model.fit(x_train,y_train, epochs=1000, batch_size=50, validation_split=0.2, verbose=1,
                 callbacks=[earlyStopping])

#4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print('RMSE: ', rmse)
print('r2: ', r2)

# print("=======================================")
# print(hist) # <keras.callbacks.History object at 0x00000195CE646850>
# print("=======================================")
# print(hist.history)
# print("=======================================")
# print(hist.history['loss'])
# print("=======================================")
# print(hist.history['val_loss'])
# print("=======================================")

# #5. draw, submit
# import matplotlib.pyplot as plt

# plt.figure(figsize=(9,6))
# plt.plot(hist.history['loss'], c='red', marker='.', label='loss')
# plt.plot(hist.history['val_loss'], c='blue', marker='.', label='val_loss')
# plt.grid()
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.title('california loss')
# plt.legend(loc='center right')
# plt.show()