import numpy as np

#1. data
x1_datasets = np.array([range(100), range(301, 401)]).transpose()
print(x1_datasets.shape)        #(100, 2) 삼성전자의 시가, 고가
x2_datasets = np.array([range(101,201), range(411,511), range(150,250)]).transpose()
print(x2_datasets.shape)        #(100, 3) 아모레의 시가, 고가, 종가
x3_datasets = np.array([range(100, 200), range(1301, 1401)]).transpose()

y1 = np.array(range(2001,2101)) #(100, ) 삼성전자의 하루 뒤 종가
y2 = np.array(range(201,301)) #(100, ) 아모레의 하루 뒤 종가
y3 = np.array(range(101,201)) #(100, ) 

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, \
y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(
    x1_datasets, x2_datasets, x3_datasets, y1, y2, y3, train_size=0.7, random_state=1234
)

print(x1_train.shape, x2_train.shape, x3_train.shape, y1_train.shape, y2_train.shape)    # (70, 2) (70, 3) (70, 2) (70,) (70,)
print(x1_test.shape, x2_test.shape, x3_test.shape, y1_test.shape, y2_test.shape)         # (30, 2) (30, 3) (30, 2) (30,) (30,)

#2. model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate, Flatten

#2-1. model1
input1 = Input(shape=(2,))
dense1 = Dense(11, activation='relu', name='ds11')(input1)
dense2 = Dense(12, activation='relu', name='ds12')(dense1)
dense3 = Dense(13, activation='relu', name='ds13')(dense2)
output1 = Dense(14, activation='relu', name='ds14')(dense3)

#2-2. model2
input2 = Input(shape=(3,))
dense21 = Dense(21, activation='linear', name='ds21')(input2)
dense22 = Dense(22, activation='linear', name='ds22')(dense21)
output2 = Dense(23, activation='linear', name='ds23')(dense22)

#2-3. model3
input3 = Input(shape=(2,))
dense31 = Dense(21, activation='linear', name='ds31')(input3)
dense32 = Dense(22, activation='linear', name='ds32')(dense31)
output3 = Dense(23, activation='linear', name='ds33')(dense32)

#2-4. model merge
from tensorflow.keras.layers import concatenate, Concatenate
# merge1 = concatenate([output1, output2, output3], name='mg1')
merge1 = Concatenate()([output1, output2, output3])
merge2 = Dense(12, activation='relu', name='mg2')(merge1)
merge3 = Dense(13, name='mg3')(merge2)
last_output = Dense(1, name='last')(merge3)

#2-5. model5 분기1
dense5 = Dense(21, activation='linear', name='ds51')(last_output)
dense5 = Dense(22, activation='linear', name='ds52')(dense5)
output5 = Dense(1, activation='linear', name='ds53')(dense5)

#2-6. model6 분기2
dense6 = Dense(21, activation='linear', name='ds61')(last_output)
dense6 = Dense(22, activation='linear', name='ds62')(dense6)
output6 = Dense(1, activation='linear', name='ds63')(dense6)

#2-7. model6 분기3
dense7 = Dense(21, activation='linear', name='ds71')(last_output)
dense7 = Dense(22, activation='linear', name='ds72')(dense7)
output7 = Dense(1, activation='linear', name='ds73')(dense7)

model = Model(inputs=[input1, input2, input3], outputs=[output5, output6, output7])

model.summary()

#3. compile, fit
model.compile(loss='mse', optimizer='adam')
model.fit([x1_train, x2_train, x3_train], [y1_train, y2_train, y3_train], epochs=100, batch_size=8)

#4. evaluate, predict
loss = model.evaluate([x1_test, x2_test, x3_test], [y1_test, y2_test, y3_test])
print('loss : ', loss)
y_predict = model.predict([x1_test, x2_test, x3_test])
print('y_predict : ', y_predict)
print('y_predict.shape : ', y_predict.shape)

"""
■ concatenate vs Concatenate
둘 다 같은 값으로 나오지만 concatenate를 사용할 때에는 단순히 concatenate([output1, output2])으로 쓰면 되고,
Concatenate를 사용할 때에는 Concatenate() 이렇게 () 괄호를 써줘야 한다.
"""
