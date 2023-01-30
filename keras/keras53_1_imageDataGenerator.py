import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

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
    target_size = (200, 200),
    batch_size = 10,
    class_mode = 'binary',
    color_mode = 'grayscale',
    shuffle = True,
    # Found 160 images belonging to 2 classes.
)

xy_test = test_datagen.flow_from_directory(
    './_data/brain/test/',
    target_size = (200, 200),
    batch_size = 100000,
    class_mode = 'binary',
    color_mode = 'grayscale',
    shuffle = True,
    # Found 120 images belonging to 2 classes.
)

print(xy_train)

from sklearn.datasets import load_boston
datasets = load_boston()
print(datasets)

# from sklearn.datasets import load_iris
# datasets = load_iris()
# print(datasets)

print(xy_train[0])
print(xy_train[0][0])
print(xy_train[0][1])
print(xy_train[0][0].shape)         # (10, 200, 200, 1)
print(xy_train[0][1].shape)         # (10,)
print(type(xy_train))               # <class 'keras.preprocessing.image.DirectoryIterator'>

print(type(xy_train))               # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))            # <class 'tuple'>
print(type(xy_train[0][0]))         # <class 'numpy.ndarray'>


"""
■ 자료
train_ad 80장, train_normal 80장 총 160장
폴더 ad, normal 순이므로 각각 0, 1로 표시한다.
"""