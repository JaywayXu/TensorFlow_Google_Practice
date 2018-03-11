import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# #### 1. 将输入转化成TFRecord格式并保存。
# 定义函数转化变量类型。
# 生成整数类型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串类型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 将数据转化为tf.train.Example格式。
def _make_example(pixels, label, image):
    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(label)),
        'image_raw': _bytes_feature(image_raw)
    }))
    return example


# 读取mnist训练数据。
mnist = input_data.read_data_sets("../../datasets/MNIST_data", dtype=tf.uint8, one_hot=True)
# 训练图片
images = mnist.train.images
# 训练数据对应的标签
labels = mnist.train.labels
# 训练图像的分辨率
pixels = images.shape[1]
num_examples = mnist.train.num_examples

# 输出包含训练数据的TFRecord文件。
with tf.python_io.TFRecordWriter("output.tfrecords") as writer:
    for index in range(num_examples):
        example = _make_example(pixels, labels[index], images[index])
        # 将一个Example 写入TFRecord文件
        writer.write(example.SerializeToString())

print("TFRecord训练文件已保存。")

# 读取mnist测试数据。
images_test = mnist.test.images
labels_test = mnist.test.labels
pixels_test = images_test.shape[1]
num_examples_test = mnist.test.num_examples

# 输出包含测试数据的TFRecord文件。
with tf.python_io.TFRecordWriter("output_test.tfrecords") as writer:
    for index in range(num_examples_test):
        example = _make_example(
            pixels_test, labels_test[index], images_test[index])
        writer.write(example.SerializeToString())
print("TFRecord测试文件已保存。")

# #### 2. 读取TFRecord文件
# 读取文件
# 创建一个reader来读取TFRecord文件中的样例
reader = tf.TFRecordReader()
# 创建一个队列来维护输入文件列表
filename_queue = tf.train.string_input_producer(["output.tfrecords"])
# 从文件中读取一个样例，如果需要解析多个样例，可以使用read_up_to函数一次性读取多个杨丽
_, serialized_example = reader.read(filename_queue)

# 解析读取的样例
# 解析读取的一个样例，如果需要解析多个样例，可以用parse_example函数
features = tf.parse_single_example(
    serialized_example,
    # Tensorflow提供两种不同的属性解析方法，一种方法是tf.FixedLenFeature,
    # 这种方法解析结果为一个Tensor，另一种方法是tf.VarLenFeature,这种方法得到的解析结果是SparseTensot,
    # 用于处理稀疏数据，这里解析数据的格式需要和上面程序写入时的格式保持一致
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    })

# tf.decode_raw可以将字符串解析成图像对应的像素数组
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()

# 启动多线程处理输入数据。
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])
