import tensorflow as tf

"""tf。QueueRunner主要用于"""
# #### 1. 定义队列及其操作
# 声明一个先进先出的队列，队列中最多有100个元素，类型为实数
queue = tf.FIFOQueue(100, "float")
# 定义队列的入队操作
enqueue_op = queue.enqueue([tf.random_normal([1])])

qr = tf.train.QueueRunner(queue, [enqueue_op]*5)
tf.train.add_queue_runner(qr)
out_tensor = queue.dequeue()

# #### 2. 启动线程。
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for _ in range(3): print(sess.run(out_tensor)[0])
    coord.request_stop()
    coord.join(threads)
