
# coding: utf-8

# In[1]:


import tensorflow as tf


# #### 1. 保存计算两个变量和的模型。

# In[2]:


v1 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))
v2 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))
result = v1 + v2

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "Saved_model/model.ckpt")


# #### 2. 加载保存了两个变量和的模型。

# In[3]:


with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model.ckpt")
    print sess.run(result)


# #### 3. 直接加载持久化的图。因为之前没有导出v3，所以这里会报错。

# In[4]:


saver = tf.train.import_meta_graph("Saved_model/model.ckpt.meta")
v3 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))

with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model.ckpt")
    print sess.run(v1) 
    print sess.run(v2) 
    print sess.run(v3) 


# #### 4. 变量重命名。

# In[5]:


v1 = tf.Variable(tf.constant(1.0, shape=[1]), name = "other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name = "other-v2")
saver = tf.train.Saver({"v1": v1, "v2": v2})

