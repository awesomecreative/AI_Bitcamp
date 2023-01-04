import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([range(10),range(21,31),range(201,211)]) # (3, 10)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]]) # (2, 10)

# [실습] train_test_split을 이용하여
# 7:3으로 잘라서 모델 구현 / 소스 완성하기

x=x.T
y=y.T

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True, random_state=123)

print('x_train : ', x_train)
print('x_test : ', x_test)       
print('y_train : ', y_train)      
print('y_test : ', y_test) 

"""
x_train :  [[  5  26 206]
 [  8  29 209]
 [  3  24 204]
 [  1  22 202]
 [  6  27 207]
 [  9  30 210]
 [  2  23 203]]
x_test :  [[  4  25 205]
 [  0  21 201]
 [  7  28 208]]
y_train :  [[ 6.   1.3]
 [ 9.   1.6]
 [ 4.   1. ]
 [ 2.   1. ]
 [ 7.   1.4]
 [10.   1.4]
 [ 3.   1. ]]
y_test :  [[5.  2. ]
 [1.  1. ]
 [8.  1.5]]
"""