
# coding: utf-8

# In[1]:


import tensorflow as tf


# #### 1. 定义变量及滑动平均类

# In[2]:


v1 = tf.Variable(0, dtype=tf.float32)
step = tf.Variable(0, trainable=False)
ema = tf.train.ExponentialMovingAverage(0.99, step)
maintain_averages_op = ema.apply([v1]) 


# #### 2. 查看不同迭代中变量取值的变化。

# In[3]:


with tf.Session() as sess:
    
    # 初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print sess.run([v1, ema.average(v1)])
    
    # 更新变量v1的取值
    sess.run(tf.assign(v1, 5))
    sess.run(maintain_averages_op)
    print sess.run([v1, ema.average(v1)]) 
    
    # 更新step和v1的取值
    sess.run(tf.assign(step, 10000))  
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_averages_op)
    print sess.run([v1, ema.average(v1)])       
    
    # 更新一次v1的滑动平均值
    sess.run(maintain_averages_op)
    print sess.run([v1, ema.average(v1)])       

