
# coding: utf-8

# In[1]:


import tensorflow as tf


# #### 1. 创建队列，并操作里面的元素。

# In[2]:


q = tf.FIFOQueue(2, "int32")
init = q.enqueue_many(([0, 10],))
x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])
with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v, _ = sess.run([x, q_inc])
        print v


# In[3]:


import numpy as np
import threading
import time


# #### 2. 这个程序每隔1秒判断是否需要停止并打印自己的ID。

# In[4]:


def MyLoop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand()<0.1:
            print "Stoping from id: %d\n" % worker_id,
            coord.request_stop()
        else:
            print "Working on id: %d\n" % worker_id, 
        time.sleep(1)


# #### 3. 创建、启动并退出线程。

# In[5]:


coord = tf.train.Coordinator()
threads = [threading.Thread(target=MyLoop, args=(coord, i, )) for i in xrange(5)]
for t in threads:t.start()
coord.join(threads)

