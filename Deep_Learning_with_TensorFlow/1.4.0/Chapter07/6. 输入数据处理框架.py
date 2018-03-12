import tensorflow as tf

'''
min_after_dequeue参数是tf.train_shuffle_batch函数特有的，
min_after_dequeue参数限制了出队时队列中元素的最小个数来保证随机打乱顺序的作用，
当出队函数被调用但是队列中元素不够时，出队操作将等待更多的元素入队才会完成。
如果min_after_dequeue参数被设定，capacity也应该相应的调整来满足性能需求。'''
# #### 1. 创建文件列表，通过文件列表创建输入文件队列，读取文件为本章第一节创建的文件。
files = tf.train.match_filenames_once("output.tfrecords")
filename_queue = tf.train.string_input_producer(files, shuffle=False)

# #### 3. 解析TFRecord文件里的数据。
# 读取文件。

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

# 解析读取的样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    })

decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
retyped_images = tf.cast(decoded_images, tf.float32)
labels = tf.cast(features['label'], tf.int32)
# pixels = tf.cast(features['pixels'],tf.int32)
images = tf.reshape(retyped_images, [784])

# #### 4. 将文件以100个为一组打包。
min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3*batch_size

image_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                  batch_size=batch_size,
                                                  capacity=capacity,
                                                  min_after_dequeue=min_after_dequeue)


# #### 5. 训练模型。
def inference(input_tensor, weights1, biases1, weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.matmul(layer1, weights2) + biases2


# 模型相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 5000

weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

y = inference(image_batch, weights1, biases1, weights2, biases2)

# 计算交叉熵及其平均值
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=label_batch)
cross_entropy_mean = tf.reduce_mean(cross_entropy)

# 损失函数的计算
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
regularaztion = regularizer(weights1) + regularizer(weights2)
loss = cross_entropy_mean + regularaztion

# 优化损失函数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 初始化会话，并开始训练过程。
with tf.Session() as sess:
    # tf.global_variables_initializer().run()
    sess.run((tf.global_variables_initializer(),
              tf.local_variables_initializer()))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 循环的训练神经网络。
    for i in range(TRAINING_STEPS):
        if i%1000 == 0:
            print("After", i, "training step(s)", "loss is  ", sess.run(loss))

        sess.run(train_step)
    coord.request_stop()
    coord.join(threads)
'''如果需要多个线程处理不同文件中的样例时，可以使用tf.train.shuffle_batch_join函数。
此函数会从输入文件队列中获取不同文件分配给不同的线程。tf.train.shuffle_batch函数和tf.train.shuffle_batch_join函数
都可以完成多线程并行的方式来进行数据预处理，但它们各有优劣。
对于tf.train.shuffle_batch函数，不同线程会读取同一个文件，如果一个文件中的用例比较相似，
比如都属于同一个类别，那么神经网络的训练效果有可能会受到影响。所以在使用tf.train.shuffle_batch函数时，需要尽量将同一个TFRecord文件中的样例随机打乱
而会用tf.train.shuffle_batch_join函数时，不同线程会读取不同的文件。如果读取数据的线程数比总文件数还大，那么多个线程可能会读取同一个文件中相近的数据。
而且多个线程读取多个文件可能导致过多的硬盘寻址，从而使得读取效率降低。不同的并行化方式各有所长，具体采用哪一种方法需要根据具体情况来确定。
'''
