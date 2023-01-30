import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

#1. data
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    vertical_flip = True,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    rotation_range = 0.5,
    zoom_range = 1.2,
    shear_range = 0.7,
    fill_mode = 'nearest',
)


test_datagen = ImageDataGenerator(
    rescale = 1./255,
)

xy_train = train_datagen.flow_from_directory(
    './_data/brain/train/',
    target_size = (100, 100),
    batch_size = 100,
    class_mode = 'binary',
    color_mode = 'grayscale',
    shuffle = True,
    # Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',
    target_size = (100, 100),
    batch_size = 100,
    class_mode = 'binary',
    color_mode = 'grayscale',
    shuffle = True,
    # Found 120 images belonging to 2 classes.
)

print(xy_train)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2,2), input_shape=(100, 100, 1)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. compile, fit
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit_generator(xy_train, steps_per_epoch=100, epochs=1000,
                    validation_data=xy_test,
                    validation_steps=4, )

accuracy = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', accuracy[-1])
print('val_acc : ', val_acc[-1])

#4. evaluate, predict



# from sklearn.datasets import load_boston
# datasets = load_boston()
# print(datasets)

# from sklearn.datasets import load_iris
# datasets = load_iris()
# print(datasets)

# print(xy_train[0])
# print(xy_train[0][0])
# print(xy_train[0][1])
# print(xy_train[0][0].shape)         # (10, 200, 200, 1)
# print(xy_train[0][1].shape)         # (10,)
# print(type(xy_train))               # <class 'keras.preprocessing.image.DirectoryIterator'>

# print(type(xy_train))               # <class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0]))            # <class 'tuple'>
# print(type(xy_train[0][0]))         # <class 'numpy.ndarray'>

"""
<학습 내용>
그림은 gpu로 돌려야 빨리 나옴.
"""
