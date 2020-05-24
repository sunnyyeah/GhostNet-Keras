# GhostNet-Keras
使用keras复现GhostNet

# 运行环境
- tensorflow1.11
- keras2.2.4

# 各个文件说明
- ghost_module.py：基本函数以及ghost模块
- ghostNet.py：ghostNet的整体结构
- train.py：训练网络
- predict.py：预测网络，求取平均准确率
- visualize.py：可视化中间层

# 训练方式
1. 修改train.py中的TrainPATH和ValPATH
2. 运行train.py

# 数据集的存放
datasets/train、datasets/val、datasets/test
