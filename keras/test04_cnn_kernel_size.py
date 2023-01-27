from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()                                                    
model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(5,5,1)))  
model.add(Conv2D(7, (2,2)))                       
model.add(Conv2D(6, 2))
model.add(Flatten())                                                         
model.add(Dense(units=10))                                                 
model.add(Dense(4, activation='relu'))                                       
model.summary()

"""
kernel_size=(2,2) 랑 그냥 (2,2) 적는 거랑 그냥 2 적는 거랑 똑같다.
"""