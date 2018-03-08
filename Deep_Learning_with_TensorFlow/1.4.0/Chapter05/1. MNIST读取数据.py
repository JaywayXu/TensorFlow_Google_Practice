# #### 在Yann LeCun教授的网站中（http://yann.lecun.com/exdb/mnist ） 对MNIST数据集做出了详细的介绍。

# #### 1. 读取数据集，第一次TensorFlow会自动下载数据集到下面的路径中。

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../../datasets/MNIST_data/", one_hot=True)

# #### 2. 数据集会自动被分成3个子集，train、validation和test。以下代码会显示数据集的大小。

print("Training data size: ", mnist.train.num_examples)
print("Validating data size: ", mnist.validation.num_examples)
print("Testing data size: ", mnist.test.num_examples)

# #### 3. 查看training数据集中某个成员的像素矩阵生成的一维数组和其属于的数字标签。

print("Example training data: ", mnist.train.images[0])
print("Example training data label: ", mnist.train.labels[0])

# #### 4. 使用mnist.train.next_batch来实现随机梯度下降。

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)  # 从train的集合中选取batch_size个训练数据。
print("X shape:", xs.shape)
print("Y shape:", ys.shape)
