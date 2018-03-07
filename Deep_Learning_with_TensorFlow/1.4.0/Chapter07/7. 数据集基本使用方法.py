
# coding: utf-8

# In[1]:


import tempfile
import tensorflow as tf


# #### 1. 从数组创建数据集。

# In[2]:


input_data = [1, 2, 3, 5, 8]
dataset = tf.data.Dataset.from_tensor_slices(input_data)

# 定义迭代器。
iterator = dataset.make_one_shot_iterator()

# get_next() 返回代表一个输入数据的张量。
x = iterator.get_next()
y = x * x

with tf.Session() as sess:
    for i in range(len(input_data)):
        print(sess.run(y))


# #### 2. 读取文本文件里的数据。

# In[3]:


# 创建文本文件作为本例的输入。
with open("./test1.txt", "w") as file:
    file.write("File1, line1.\n") 
    file.write("File1, line2.\n")
with open("./test2.txt", "w") as file:
    file.write("File2, line1.\n") 
    file.write("File2, line2.\n")

# 从文本文件创建数据集。这里可以提供多个文件。
input_files = ["./test1.txt", "./test2.txt"]
dataset = tf.data.TextLineDataset(input_files)

# 定义迭代器。
iterator = dataset.make_one_shot_iterator()

# 这里get_next()返回一个字符串类型的张量，代表文件中的一行。
x = iterator.get_next()  
with tf.Session() as sess:
    for i in range(4):
        print(sess.run(x))


# #### 3. 解析TFRecord文件里的数据。读取文件为本章第一节创建的文件。

# In[4]:


# 解析一个TFRecord的方法。
def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image_raw':tf.FixedLenFeature([],tf.string),
            'pixels':tf.FixedLenFeature([],tf.int64),
            'label':tf.FixedLenFeature([],tf.int64)
        })
    decoded_images = tf.decode_raw(features['image_raw'],tf.uint8)
    retyped_images = tf.cast(decoded_images, tf.float32)
    images = tf.reshape(retyped_images, [784])
    labels = tf.cast(features['label'],tf.int32)
    #pixels = tf.cast(features['pixels'],tf.int32)
    return images, labels

# 从TFRecord文件创建数据集。这里可以提供多个文件。
input_files = ["output.tfrecords"]
dataset = tf.data.TFRecordDataset(input_files)

# map()函数表示对数据集中的每一条数据进行调用解析方法。
dataset = dataset.map(parser)

# 定义遍历数据集的迭代器。
iterator = dataset.make_one_shot_iterator()

# 读取数据，可用于进一步计算
image, label = iterator.get_next()

with tf.Session() as sess:
    for i in range(10):
        x, y = sess.run([image, label]) 
        print(y)


# #### 4. 使用initializable_iterator来动态初始化数据集。

# In[5]:


# 从TFRecord文件创建数据集，具体文件路径是一个placeholder，稍后再提供具体路径。
input_files = tf.placeholder(tf.string)
dataset = tf.data.TFRecordDataset(input_files)
dataset = dataset.map(parser)

# 定义遍历dataset的initializable_iterator。
iterator = dataset.make_initializable_iterator()
image, label = iterator.get_next()

with tf.Session() as sess:
    # 首先初始化iterator，并给出input_files的值。
    sess.run(iterator.initializer,
             feed_dict={input_files: ["output.tfrecords"]})
    # 遍历所有数据一个epoch。当遍历结束时，程序会抛出OutOfRangeError。
    while True:
        try:
            x, y = sess.run([image, label])
        except tf.errors.OutOfRangeError:
            break 

