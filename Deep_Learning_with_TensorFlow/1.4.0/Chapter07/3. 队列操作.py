import tensorflow as tf

# #### 1. 创建队列，并操作里面的元素。

# 创建一个先进先出的队列，指定队列最多可以保存两个元素，并指定类型为整数
q = tf.FIFOQueue(2, "int32")
init = q.enqueue_many(([0, 10],))
x = q.dequeue()
# 使用dequence函数将队列中第一个元素出队列，这个元素的值将被保存在变量x中
y = x + 1
# 将x+1后的值重新加入队列
q_inc = q.enqueue([y])
with tf.Session() as sess:
    init.run()
    for _ in range(5):
        # 运行q_inc将执行数据出队列，出队的元素+1，重新加入队列的整个过程
        v, _ = sess.run([x, q_inc])
        print(v)  # 打印出队元素的取值

# 0
# 10
# 1
# 11
# 2
import numpy as np
import threading
import time


# #### 2. 这个程序每隔1秒判断是否需要停止并打印自己的ID。
def MyLoop(coord, worker_id):
    # 使用tf.Coordinator类提供的协同工具判断当前线程是否需要停止
    while not coord.should_stop():
        # 随机停止所有线程
        if np.random.rand() < 0.1:
            print("Stoping from id: %d\n"%worker_id)
            # 调用coord.request_stop()函数来通知其他线程停止
            coord.request_stop()
        else:
            # 打印当前线程ID
            print("Working on id: %d\n"%worker_id)
        time.sleep(1)


# #### 3. 创建、启动并退出线程
# 声明一个tf.train.Coordinator类来协同多个线程
coord = tf.train.Coordinator()
# 声明创建5个线程
threads = [threading.Thread(target=MyLoop, args=(coord, i,)) for i in range(5)]
# 启动所有线程
for t in threads: t.start()
# 等待所有线程退出
coord.join(threads)

# Working on id: 0
#
# Working on id: 1
#
# Working on id: 2
#
# Working on id: 3
#
# Working on id: 4
#
# Working on id: 0
#
# Working on id: 2
#
# Working on id: 3
# Working on id: 1
# Working on id: 4
#
# Working on id: 0
#
# Working on id: 1
#
# Working on id: 4
#
# Stoping from id: 3
