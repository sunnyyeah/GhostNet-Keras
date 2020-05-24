from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras import backend as K
from ghostNet import GhostNet

import numpy as np

import matplotlib.pyplot as plt
import cv2
import math

def slices(dw, n, data_format='channels_last'):
    if data_format == 'channels_last':
        return dw[:, :, :, :n]
    else:
        return dw[:, :n, :, :]


# model = load_model("./checkpoint/GhostNet-100epoch_my-datasets224_pretrainModel.hdf5",
#                   custom_objects={"slices": slices})

model = GhostNet((224, 224, 3), 25).build()
model.load_weights("./pretrain_model/ghostNet_my-datasets224_-100.h5")

for i, layer in enumerate(model.layers):
    print(i, layer.name)

print(model.layers[0].name)
print(model.layers[1].name)

img_path = "airplane.jpg"
src = cv2.imread(img_path)
cv2.imshow("src", src)
# cv2.waitKey(0)

# 加载图片
img = image.load_img(img_path, target_size=(224, 224))  # target_size为图片的尺寸

# 将图片转为数组表示
x = image.img_to_array(img)     # x.shape = (224,224,3)
x = np.expand_dims(x, axis=0)   # 扩展维度为0的数据量，x.shape = (1,224,224,3)
x = preprocess_input(x)

# 设置可视化的层(这里的层指的是网络架构图中的每个框框，也就是一个操作算一层)
# 根据代码中的网络结构其输出层分别为3,15,34,46,72,91,110,122,134,146,169,188,214,226,245,257,276
layer1 = K.function([model.layers[0].input], [model.layers[34].output])
f1 = layer1([x])[0]

print(f1.shape)
n = f1.shape[-1] // 5 + 1
h = f1.shape[1]
w = f1.shape[2]
print("n = ", n)
for i in range(f1.shape[-1]):       # f1.shape[-1]: 特征图个数
    print("绘制:", i)
    show_img = f1[0, :, :, i]       # 获取第i个通道的特征图
    # show_img.shape = [h, w]
    plt.subplot(n, 5, 1+i)
    plt.imshow(show_img)
    plt.axis('off')
   
    cv2.imshow("img", show_img)
    #cv2.waitKey(0)
#plt.savefig("./images/vis34.png")
plt.show()