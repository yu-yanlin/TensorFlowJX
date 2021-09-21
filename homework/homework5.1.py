# 导包
import tensorflow as tf
from tensorflow import keras

# 加载数据
fashion_mnist = keras.datasets.fashion_mnist
# 进行训练集 和 测试集的划分
(train_images, train_labels), (test_image, test_labels) = fashion_mnist.load_data()

# 数据预处理 将每个像素进行压缩在 0 到 1 之间
train_images = train_images / 255.0
test_image = test_image / 255.0


def create_model():
    # 创建神经网络模型
    model = tf.keras.models.Sequential([
        # 输入层
        keras.layers.Flatten(input_shape=(28, 28)),
        # 全链接神经网络
        # 隐藏层
        keras.layers.Dense(128, activation="relu"),
        # 输出
        keras.layers.Dense(10)
    ])

    # 编译模型
    model.compile(optimizer='adam',
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.metrics.SparseTopKCategoricalAccuracy()])
    # 返回模型
    return model


# 构建模型
new_model = create_model()
# 训练模型
new_model.fit(train_images,train_labels,epochs=30)
# 保存模型
new_model.save("model/homework5.h5")

