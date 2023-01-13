from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()                                                    # (60000, 5, 5, 1) : input : (N, 5, 5, 1)
model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(5,5,1)))   # (4,4,10) : (N, 4, 4, 10)
model.add(Conv2D(5, (2,2)))                         # (3,3,5) : (N, 3, 3, 5)
# model.add(Conv2D(filters=5, kernel_size=(2,2)))
# model.add(Conv2D(5, (2,2))) 이렇게 간단히 표현 가능하다.
model.add(Flatten())                                                          # (45,) : (N, 45)
model.add(Dense(units=10))                                                    # (N, 10)
model.add(Dense(4, activation='relu'))                                        # (N, 1)
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
input_shape=(5,5,1) : (5,5) 크기의 흑백 이미지를 갖고 있다.
:사진의 세로길이, 가로길이를 몇 칸으로 나눌지 직접 정해 적고, 마지막 값은 사진이 흑백이면 1 컬러면 3을 적는다.
컬러는 애초에 3장 필요하므로 input_shape의 마지막 값이 3이여야 함.

Flatten() 하면 그 전의 Conv2D의 shape 값을 곱한 만큼 column이 생겨서 연산하기 쉬운 형태로 만듬.
Flatten 한 이후에야 Dense 레이어 층에 넣어 인공 신경망을 돌릴 수 있음.
예를 들어, 그 전의 Conv2D의 shape이 (None,3,4,5)이면 Flatten하면 shape이 3x4x5=60 으로 (None,60)이 된다.

실제로 (60000, 5, 5, 1) 이런 식으로 인풋함. 60000장, 세로5, 가로5, 흑백이미지를 인풋한다는 의미이다.
행무시 열우선이기 때문에 실제 (60000, 5, 5, 1) 이미지를 (None, 5, 5, 1)로 표현한다. None은 데이터의 개수를 의미한다.

<Output Shape 계산법>
무조건 마지막 값은 filters에서 결정된다. input_shape의 마지막 값과는 상관없다.
input_shape=(x,y,k) 을 filters=e, kernel_size=(m,n) 로 통과시키면
output_shape=(x-m+1, y-n+1, e)가 된다.

<Param # 계산법>
Conv2d : (number of filters * filter height * filter width * number of input channels) + (number of filters)
Dense : (number of input neurons * number of output neurons) + (number of output neurons)
number of input channels 는 RGB image 면 3, grayscale image 면 1 임.

model.add(Conv2D(filters=10, kernel_size=(2,2), input_shape=(5,5,1)))   # param : 10*2*2*1+10 = 50
model.add(Conv2D(filters=5, kernel_size=(2,2)))                         # param : 5*2*2*10+5 = 205

# model.add(Conv2D(filters=5, kernel_size=(2,2)))
# model.add(Conv2D(5, (2,2))) 이렇게 간단히 표현 가능하다.
"""

"""
<Conv2D의 설명>

<Arguments>
1) filters : convolutional layer(합성곱 층)에서 사용하는 필터의 수 (dense에서 output layer의 노드 수랑 비슷한 의미)
Each filter is a small matrix that is convolved with the input data to extract features.
More filters mean more features are extracted from the input data, and this results in more complex models.
2) kernel_size : 합성곱 층에서 사용하는 필터의 크기
typically represented as a tuple of two integers, such as (3, 3)
3) strides : 합성곱 실행에서 사용되는 단계의 크기
typically represented as a tuple of two integers, such as (1, 1)
(1,1)은 필터가 행, 열 방향으로 한 번에 한 픽셀씩 이동한다는 것을 의미함.
4) padding : 합성곱 실행하기 전에 input data에 사용되는 패딩
valid : padding 적용X , same : padding 적용O
합성곱 연산 후 원본 이미지의 공간 차원을 유지하기 위해 미리 이미지 가장자리 주변에 픽셀의 추가 레이어를 추가하는 과정을 말함.
이렇게 하면 출력 이미지가 입력 이미지보다 작아지는 것을 방지하고 원본 이미지의 공간적 특징을 보존하는 데도 도움이 됨.
5) data_foramt : channels_last(default), channels_first
아래 Input shape과 output shape 참고하기
6) dilation_rate : 더 큰 시야를 가지는 필터를 만들기 위한 확장률
7) groups : 입력과 출력 사이의 연결을 제어함.
8) activation : 합성곱 실행 후에 사용되는 최적화 함수 (비선형성을 주기 위함임)
relu, sigmoid, tanh, elu 등이 있음.
9) use_bias : convolution layer 에 bias vector 포함 여부 결정. True가 기본값임.
10) kernel_initializer, bias_initializer : 합성곱 층의 가중치와 편향의 초기 값 설정 (중요함: 훈련시 네트워크 학습 방법 결정함.)
11) kernel_regularizer, bias_regularizer : 가중치 정교화
12) kernel_constraint, bias_constraint : 제약 조건 지정
13) activity_regularizer : 활동 정규화

<Input shape : 4-dimensional tensor>
(batch_size, height, width, channels) : data_format='channels_last' (Tensorflow에서는 이게 default)
(batch_size, channels, height, width) : data_format='channels_first' (Theano library에서는 이게 default)
라이브러리나 프레임워크에 따라 데이터 형태의 기본값이 다를 수 있으므로 유의해야한다.
batch_size : 데이터의 샘플 수 : 훈련의 수
height, width : 세로, 가로 (행, 열)
channels : 인풋 데이터의 채널수 (RGB는 3, grayscale은 1)

<output shape : 4-dimensional tensor>
(batch_size, new_height, new_width, number_of_filters).
new_height, new_width : stride, padding and kernel size와 같은 convolutional layer에 따라 달라짐.
"""

"""
<Dense의 설명>

<Arguments>
1) units: 레이어의 뉴런수(노드수)
2) activation: 최적화 함수
3) use_bias: bias vector 사용 여부
4) kernel_initializer : 합성곱 층의 가중치 초기 값 설정
5) bias_initializer: 편향의 초기 값 설정
6) kernel_regularizer: 합성곱 층 가중치 정교화
7) bias_regularizer: 편향 벡터 정교화
8) activity_regularizer: output layer에 적용되는 정교화 함수
9) kernel_constraint: 제약 조건 지정
10) bias_constraint: 제약 조건 지정

<Input shape>
(batch_size, ... , input_dim) N차원 텐서

<Output shape>
(batch_size, ... , units) N차원 텐서
"""
# Boolean : 참과 거짓을 의미하는 데이터 타입
# keras parameter 궁금하다면 keras.io 홈페이지 검색하면 된다.