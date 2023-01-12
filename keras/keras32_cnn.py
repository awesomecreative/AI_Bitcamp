from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(5,5,1)))
model.add(Conv2D(filters=5, kernel_size=(2,2)))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))
model.summary()


"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 4, 4, 10)          50

 conv2d_1 (Conv2D)           (None, 3, 3, 5)           205

 flatten (Flatten)           (None, 45)                0

 dense (Dense)               (None, 10)                460

 dense_1 (Dense)             (None, 1)                 11

=================================================================
Total params: 726
Trainable params: 726
Non-trainable params: 0
_________________________________________________________________

"""

"""
<CNN에서 이미지를 인식하는 방법>
CNN : Convolutional Neural Networks
한 레이어 지날 때마다 이미지를 조각내서 행렬 형태 데이터로 만들어 높을 특성을 계속 합치고 낮은 특성을 도태시킨다.
나중에 나온 행렬 결과값을 가지고 이미지를 인식한다.

filters=10 : 사진 1장의 필터를 10판으로 늘리겠다는 의미이다. (연산량 증가)
kernel_size=(2,2) : 연산할 때 사진을 자르는 단위 (행, 열)
input_shape=(5,5,1) : (5,5) 크기의 1개 이미지를 갖고 있다.
:사진의 세로길이, 가로길이를 몇 칸으로 나눌지 직접 정해 적고, 마지막 값은 사진이 흑백이면 1 컬러면 3을 적는다.
컬러는 애초에 3장 필요하므로 input_shape의 마지막 값이 3이여야 함.

Flatten() 하면 그 전의 Conv2D의 shape 값을 곱한 만큼 column이 생겨서 연산하기 쉬운 형태로 만듬.
Flatten 한 이후에야 Dense 레이어 층에 넣어 인공 신경망을 돌릴 수 있음.
예를 들어, 그 전의 Conv2D의 shape이 (None,3,4,5)이면 Flatten하면 shape이 3x4x5=60 으로 (None,60)이 된다.

<Output Shape 계산법>
무조건 마지막 값은 filters에서 결정된다. input_shape의 마지막 값과는 상관없다.
input_shape=(x,y,k) 을 filters=e, kernel_size=(m,n) 로 통과시키면
output_shape=(x-m+1, y-n+1, e)가 된다.

<질문!!!!!!!!!!!!!!!!>
Q) input_shape의 마지막 값이 3이면 3장 있다는 것이고, filters=10 하면 1장당 10판 해서 총 30일 줄 알았는데 아님...
왜 input_shape의 마지막 값과는 상관이 없는지 궁금함.
"""
