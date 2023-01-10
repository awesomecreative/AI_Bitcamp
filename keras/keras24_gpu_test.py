import tensorflow as tf
print(tf.__version__) #2.7.4

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

if(gpus):
    print("gpu 실행중")
else:
    print("gpu 실행안함")

"""
<가상환경 tf27 (cpu만 있는 것)>
2.7.4
[]
gpu 실행안함

<가상환경 tf274gpu로 바꾸면>
2.7.4
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
gpu 실행중
"""