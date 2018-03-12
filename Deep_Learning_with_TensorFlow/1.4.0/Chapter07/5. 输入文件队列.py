import tensorflow as tf


# #### 1. 生成文件存储样例数据。
# 创建TFRecord文件的帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


num_shards = 2  # 定义总共写入多少文件
instances_per_shard = 2  # 定义每个文件写入有多少个数据
for i in range(num_shards):
    # 将数据分为多个文件时，可以将不同文件以类似0000n-of-0000m的后缀加一区分。其中m表示了数据总共被存在了多少文件中，n表示当前文件的编号
    # 式样的方式既方便了通过正则表达式获取文件列表，又在文件名中加入了更多信息
    filename = ('data.tfrecords-%.5d-of-%.5d'%(i, num_shards))
    # 将Example结构写入TFRecord文件。
    writer = tf.python_io.TFRecordWriter(filename)
    # 将数据封装成Example结构并写入TFRecord文件中
    for j in range(instances_per_shard):
        # Example结构仅包含当前样例属于第几个文件以及是当前文件的第几个样本。
        example = tf.train.Example(features=tf.train.Features(feature={
            'i': _int64_feature(i),
            'j': _int64_feature(j)}))
        writer.write(example.SerializeToString())
    writer.close()

# #### 2. 读取文件。
# 使用tf.train.match_filenames_once函数获取文件列表
files = tf.train.match_filenames_once("data.tfrecords-*")

# 通过tf.train.string_input_producer函数创建输入队列，输入队列的文件列表为
# tf.train.match_filenames_once函数获取的文件列表
filename_queue = tf.train.string_input_producer(files, shuffle=False)
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64),
    })
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    print(sess.run(files))
    # 声明tf.train.Coordinator类来协同不同线程，并启动线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 多次获取数据
    for i in range(6):
        print(sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)

# #### 3. 组合训练数据（Batching）
"""将多个输入样例组织成一个batch可以提高模型的训练的效率，所以在得到单个的样例的预处理结果之后，还需要将其组织成batch,然后在提供给神经网络的输入层
Tensorflow提供了tf.train.batch和tf.train.shuffle_batch函数来将单个的样例组织成batch的形式输出。唯一的区别是后者会将数据的顺序打乱"""
example, label = features['i'], features['j']
batch_size = 2
capacity = 1000 + 3*batch_size
example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(3):
        cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
        print(cur_example_batch, cur_label_batch)
    coord.request_stop()
    coord.join(threads)

