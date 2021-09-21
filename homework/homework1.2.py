import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

# 加载数据
fashion_mnist = keras.datasets.fashion_mnist
# 进行训练集 和 测试集的划分
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 对刚才的模型进行测试
# 构建标签列表
class_names = ['T恤/上衣', '裤子', '套衫', '连衣裙', '外套','凉鞋', '衬衫', '运动鞋', '手提包', '踝靴']

# 数据预处理
train_images = train_images / 255.0
test_images = test_images / 255.0

# 加载神经网络模型
new_model = keras.models.load_model('model/homework1.h5')

# 对刚才的训练模型进行测试
test_loss,test_acc = new_model.evaluate(test_images, test_labels, verbose=2)

# 对数据进行预处理
probability_model = tf.keras.Sequential([new_model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
# 第15个数据 要与之后的 位置对应
print(predictions[15])

# 打印所有测试结果
# 对图像进行显示
# 显示中文
plt.rcParams['font.sans-serif'] = ['FangSong']
plt.figure()
plt.subplot(1,2,1)
plt.xlabel([])
plt.ylabel([])
plt.imshow(test_images[15],cmap=plt.cm.binary)
plt.xlabel("{} 预测正确率：{:2.0f}%".format(class_names[np.argmax(predictions[15])],
                                      100*np.max(predictions[15])),fontsize=20,color='blue')
plt.subplot(1,2,2)
plt.xticks(range(10),class_names)
plt.ylabel([])
thisplot = plt.bar(range(10), predictions[15], color='#777777')
plt.ylim([0, 1])
predicted_label = np.argmax(predictions[15])

thisplot[predicted_label].set_color('blue')

plt.show()
print('模型预测的结果为：{}'.format(class_names[np.argmax(predictions[15])]))