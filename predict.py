from keras.preprocessing import image
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input

import numpy as np
import time
import os
from ghostNet import GhostNet


def slices(dw, n, data_format='channels_last'):
    if data_format == 'channels_last':
        return dw[:, :, :, :n]
    else:
        return dw[:, :n, :, :]


my_datasets = ['airplane', 'airport', 'basketball_court', 'beach', 'bridge', 'commercial_area',
               'farmland', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area',
               'intersection', 'mountain', 'overpass', 'parking_lot', 'railway', 'railway_station',
               'residential', 'river', 'runway', 'ship', 'stadium', 'storage_tank', 'tennis_court']

# rootdir下面是类别文件,类别文件下面才是相关的图片
def readImage2(rootdir):
    ClassList = os.listdir(rootdir)
    # print(imageName)

    # model = load_model(LoadModel, custom_objects={"slices": slices})
    model = GhostNet((224, 224, 3), 25).build()
    model.load_weights("./pretrain_model/ghostNet_my-datasets224_-100.h5")
    # f = open("preds-50.txt", 'a')      # 打开preds.txt文件夹
    # f.truncate()                    # 清空文件夹
    RightNum = 0    # 预测正确图片总量
    imageNum = 0    # 图片总量
    alltime = 0     # 总时间
    for ClassName in ClassList:
        eachCorrect = 0
        eachNum = len(os.listdir(rootdir + ClassName))
        imageNum = imageNum + eachNum
        # print(imageNum)
        for i in os.listdir(rootdir + ClassName):
            # 加载图片
            # img_path = rootdir + 'stadium_0' + str(i) + '.jpg'
            img_path = rootdir + ClassName + os.sep + i
            img = image.load_img(img_path, target_size=(224, 224))  # target_size为图片的尺寸

            # 将图片转为数组表示
            x = image.img_to_array(img)     # x.shape = (224,224,3)
            x = np.expand_dims(x, axis=0)   # 扩展维度为0的数据量，x.shape = (1,224,224,3)
            x = preprocess_input(x)

            # 计算每张图的预测时间：start_time1为起始时间，end_time1为终止时间
            # <time>
            start_time1 = time.time()

            # 预测
            preds = model.predict(x)  # 获取每个类别的训练概率，变量类型为ndarray
            result = preds.tolist()  # 将变量类型转为list
            # print(max(result[0]))                         # 输出每个类型的预测值
            labels = result[0].index(max(result[0]))  # 得到概率最大的值的索引
            # print(labels)                           # 输出概率最大的值的索引
            # if i[:-9] == 'airplane':
            #     print("预测类别：", my_datasets[labels])  # 输出概率最大的类别
            #     print("真实类别：", ClassName)
            #     errorNum = 0
            #     if my_datasets[labels] != ClassName:
            #         errorNum = errorNum + 1

            end_time1 = time.time()
            alltime = alltime + end_time1 - start_time1
            # print("time:%fs" % (end_time1 - start_time1))
            # </time>

            # 计算预测正确的图片那数目
            if my_datasets[labels] == ClassName:
                RightNum = RightNum + 1         # 总的正确预测的数量
                eachCorrect = eachCorrect + 1   # 单个类别正确预测的数量

            # 使用opencv实现在图片上加文字
            # showimg = cv2.imread(img_path)
            # cv2.putText(showimg, NWPU_LABELS[labels], (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.7, (77, 255, 9), 1, cv2.LINE_AA)
            # cv2.putText(showimg, i[:-8], (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.7, (9, 77, 255), 1, cv2.LINE_AA)
            # cv2.imshow("result", showimg)
            # cv2.imwrite("result.jpg", showimg)
            # cv2.waitKey(0)

            # print("\n")

            # 使用PIL添加文字
            # showimg = Image.open(img_path)
            # draw = ImageDraw.Draw(showimg)
            # draw.text((10,20), NWPU_LABELS[labels], fill = (77, 255, 9))
            # showimg.show()
        print(ClassName,"的准确率为:", eachCorrect / eachNum)

        # 将每类图片的正确率保存到.txt中
        # f.write(ClassName + ':' + str(eachCorrect / eachNum))
        # f.write('\n')

    print("\n测试图片正确率：", RightNum / imageNum)
    print("测试图片的平均时间:", alltime / imageNum)

if __name__ == "__main__":
    # readImage1()

    LoadModel = 'GhostNet-100epoch_my-datasets224_pretrainModel.hdf5'

    rootdir = '/home/wby/DeepLearning/datasets/my_datasets224/test/'

    readImage2(rootdir)
