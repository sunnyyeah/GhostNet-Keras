import tensorflow as tf
from ghostNet import GhostNet

from keras_preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.models import load_model

import time
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Conv2D,Reshape,GlobalAveragePooling2D
from keras.models import Model
from keras import backend as K

print(f"Tensorflow version: {tf.__version__}")

# ---------------------网络结构搭建------------------------- #

# 配置
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

# 加载模型
model = load_model('GhostNet-100epoch_my-datasets224_pretrainModel.hdf5')

# 超参数
TrainPATH = '/home/wby/DeepLearning/datasets/my_datasets224/train/'  # 数据集路径 ———— 针对不同的数据集有不同的值
ValPATH = '/home/wby/DeepLearning/datasets/my_datasets224/val/'
BATCH_SIZE = 16
# TEST_SIZE = 0.3     # ImageDataGenerator生成的验证集是前30%个
EPOCHS = 100
CLASSES = 25       # 数据集类别 ———— 针对不同的数据集有不同的值

# -----------------读取数据（参考keras文档理解以下函数的作用）------------------ #
# 数据处理
# datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
#                              rotation_range=10,
#                              horizontal_flip=True,
#                              validation_split=TEST_SIZE)
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=10,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                 rotation_range=10,
                                 horizontal_flip=True)

# 训练数据
print("生成的训练样本数：")
train_generator = train_datagen.flow_from_directory(TrainPATH, target_size=(224, 224),
                                                    classes=None, batch_size=BATCH_SIZE)
# 验证数据
print("生成的验证样本数：")
validation_generator = val_datagen.flow_from_directory(ValPATH, target_size=(224, 224),
                                                       classes=None, batch_size=BATCH_SIZE)


# -----------------------------------模型训练-------------------------------------- #
# 模型编译
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# train
print("steps_per_epoch = ", train_generator.samples // BATCH_SIZE)
print("validation_steps = ", validation_generator.samples // BATCH_SIZE)
result = model.fit_generator(generator=train_generator,
                             steps_per_epoch=train_generator.samples // BATCH_SIZE,     # 训练样本数/batch_size
                             epochs=EPOCHS,
                             verbose=1,
                             validation_data=validation_generator,
                             validation_steps=validation_generator.samples // BATCH_SIZE,   # 验证样本数/batch_size
                             shuffle=True)

# 保存训练的参数
model.save_weights(f"./pretrain_model/ghostNet_my-datasets224_-{EPOCHS}.h5")


# -----------------------------------可视化模型训练曲线------------------------------- #
plt.figure()
plt.plot(result.epoch, result.history['acc'], label="acc")
plt.plot(result.epoch, result.history['val_acc'], label="val_acc")
plt.scatter(result.epoch, result.history['acc'], marker='*')
plt.scatter(result.epoch, result.history['val_acc'])
plt.legend(loc='under right')
plt.show()

plt.figure()
plt.plot(result.epoch, result.history['loss'], label="loss")
plt.plot(result.epoch, result.history['val_loss'], label="val_loss")
plt.scatter(result.epoch, result.history['loss'], marker='*')
plt.scatter(result.epoch, result.history['val_loss'])
plt.legend(loc='upper right')
plt.show()

