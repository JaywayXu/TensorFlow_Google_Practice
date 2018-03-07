
# coding: utf-8

# #### 1. 假设我们输入矩阵
# 
# $
# M=\left(\begin{array}{c}
# 1&-1&0\\
# -1&2&1\\
# 0&2&-2
# \end{array}\right)
# $

# In[1]:


import tensorflow as tf
import numpy as np

M = np.array([
        [[1],[-1],[0]],
        [[-1],[2],[1]],
        [[0],[2],[-2]]
    ])

print "Matrix shape is: ",M.shape


# #### 2. 定义卷积过滤器, 深度为1。
# $
# W=\left(\begin{array}{c}
# 1&-1\\
# 0&2
# \end{array}\right)
# $

# In[2]:


filter_weight = tf.get_variable('weights', [2, 2, 1, 1], initializer = tf.constant_initializer([
                                                                        [1, -1],
                                                                        [0, 2]]))
biases = tf.get_variable('biases', [1], initializer = tf.constant_initializer(1))


# #### 3. 调整输入的格式符合TensorFlow的要求。

# In[3]:


M = np.asarray(M, dtype='float32')
M = M.reshape(1, 3, 3, 1)


# #### 4.  计算矩阵通过卷积层过滤器和池化层过滤器计算后的结果。

# In[4]:


x = tf.placeholder('float32', [1, None, None, 1])
conv = tf.nn.conv2d(x, filter_weight, strides = [1, 2, 2, 1], padding = 'SAME')
bias = tf.nn.bias_add(conv, biases)
pool = tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    convoluted_M = sess.run(bias,feed_dict={x:M})
    pooled_M = sess.run(pool,feed_dict={x:M})
    
    print "convoluted_M: \n", convoluted_M
    print "pooled_M: \n", pooled_M

