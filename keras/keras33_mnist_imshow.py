import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,) (60000, 28, 28)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(x_train[0]) # x_train의 0번째
print(y_train[0]) # y_train의 0번째

import matplotlib.pyplot as plt
plt.imshow(x_train[1000], 'gray')
plt.show()

"""
처음 데이터셋을 다운받을 때에는 다음과 같은 메시지가 뜬다.
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

shape 찍었을 때 (60000, 28, 28)은 (60000, 28, 28, 1)과 같다.
Flatten 할 때 데이터의 내용과 순서는 바뀌지 않는다.
"""