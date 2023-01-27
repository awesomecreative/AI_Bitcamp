import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#1. data
a = np.array(range(1, 101))         # [1,2,3,4,5,6, ... ,97, 98, 99, 100]
x_predict = np.array(range(96, 106))    # [96,97,98,...,102,103,104,105]
# 예상 y = 100~106 

timesteps = 5                       # x는 4개, y는 1개

def split_x(dataset, timesteps) :
    aaa = []
    for i in range(len(dataset) - timesteps +1):
        subset = dataset[i : (i+timesteps)]
        aaa.append(subset)
    return np.array(aaa)

a = split_x(a, timesteps)
x = a[:, :-1]
y = a[:, -1]

x_predict = split_x(x_predict, 4)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, shuffle=True, random_state=1)

print(x_train.shape, x_test.shape, x_predict.shape)      # (72, 4) (24, 4) (7, 4)

#2. model
model = Sequential()
model.add(Dense(256, input_shape=(4, ), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

#3. compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=3000)

#4. evaluate, predict
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_predict)
print('y_predict : ', y_predict)

"""
<1>
Epoch 3000/3000
3/3 [==============================] - 0s 3ms/step - loss: 7.6045e-05
1/1 [==============================] - 0s 82ms/step - loss: 2.0301e-04
loss :  0.00020301116455812007
y_predict :  [[100.02358 ] [101.02386 ] [102.024185] [103.024536] [104.0249  ] [105.025276] [106.02564 ]]

<2>
Epoch 3000/3000
3/3 [==============================] - 0s 3ms/step - loss: 8.5780e-05
1/1 [==============================] - 0s 83ms/step - loss: 1.1221e-04
loss :  0.00011220993474125862
y_predict :  [[100.01919 ] [101.019424] [102.01966 ] [103.019905] [104.02016 ] [105.02039 ] [106.02065 ]]
 
<3>
Epoch 3000/3000
3/3 [==============================] - 0s 5ms/step - loss: 2.2772e-05
1/1 [==============================] - 0s 75ms/step - loss: 5.8134e-05
loss :  5.813421739730984e-05
y_predict :  [[100.014824] [101.01508 ] [102.01537 ] [103.01566 ] [104.01595 ] [105.01625 ] [106.01655 ]]

★ 결론
RNN이나 DNN이나 비슷한 것 같다.
"""