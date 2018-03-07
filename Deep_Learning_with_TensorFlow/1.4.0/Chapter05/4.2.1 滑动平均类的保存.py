
# coding: utf-8

# In[1]:


import tensorflow as tf


# #### 1. 使用滑动平均。

# In[2]:


v = tf.Variable(0, dtype=tf.float32, name="v")
for variables in tf.global_variables(): print variables.name
    
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
for variables in tf.global_variables(): print variables.name


# #### 2. 保存滑动平均模型。

# In[3]:


saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    
    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    # 保存的时候会将v:0  v/ExponentialMovingAverage:0这两个变量都存下来。
    saver.save(sess, "Saved_model/model2.ckpt")
    print sess.run([v, ema.average(v)])


# #### 3. 加载滑动平均模型。

# In[4]:


v = tf.Variable(0, dtype=tf.float32, name="v")

# 通过变量重命名将原来变量v的滑动平均值直接赋值给v。
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model2.ckpt")
    print sess.run(v)

