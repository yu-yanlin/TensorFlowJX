# 导包
import tensorflow as tf
from tensorflow import keras

# 加载数据
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels),(tests_images, test_labels) = fashion_mnist.load_data()

# 数据处理  将每个像素点压缩在 0 到 1 之间
train_images = train_images / 255.0
tests_images = tests_images / 255.0

# 搭建简单神经网络
def create_model():
    # 创建神经网络模型
    model = tf.keras.models.Sequential([
        # 输入层
        keras.layers.Flatten(input_shape=(28, 28)),
        # 全链接层
        keras.layers.Dense(128, activation='relu'),
        # 输入
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
new_model.fit(train_images,tests_images,epochs=30)
# 保存模型
new_model.save("model/homework1.h5")

