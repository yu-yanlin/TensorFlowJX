'''
    流程：  1、确定目标
           2、准备数据（爬虫，下载，自制）
           3、标注
           4、搭建神经网络
           5、数据预处理（大小、格式（txt，yaml，xml））
           6、保存模型
           7、调用模型
           8、测试模型
           9、布属（web，Android，硬件）
           10、测试
           11、发布
'''

# 加载模块
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as plt

# 加载数据
fashion_mnist = keras.datasets.fashion_mnist
# 进行训练集 和 测试集的划分
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 数据预处理  将每个像素点压缩在 0 到 1 之间
train_images = train_images/255.0
test_images = test_images/255.0


# 搭建简单的神经网络
def create_model():
    # 创建神经网络模型
    model = tf.keras.models.Sequential([
        # 输入层
        keras.layers.Flatten(input_shape=(28, 28)),
        # 全链接神经网络
        # 隐藏层
        keras.layers.Dense(128, activation='relu'),
        # 输出
        keras.layers.Dense(10)
    ])

    # 编译模型
    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.metrics.SparseTopKCategoricalAccuracy()]
                  )
    # 返回模型
    return model


# 构建模型
new_model = create_model()

# 训练模型
new_model.fit(train_images,train_labels,epochs=30)

# 保存模型
new_model.save("model/my_model3.h5")