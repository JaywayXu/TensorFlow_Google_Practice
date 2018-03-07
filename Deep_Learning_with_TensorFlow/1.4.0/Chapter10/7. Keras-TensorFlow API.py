
# coding: utf-8

# ### 1. 模型定义。

# In[1]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist_data = input_data.read_data_sets('../../datasets/MNIST_data', one_hot=True)

# 通过TensorFlow中的placeholder定义输入。
x = tf.placeholder(tf.float32, shape=(None, 784))
y_ = tf.placeholder(tf.float32, shape=(None, 10))

net = tf.keras.layers.Dense(500, activation='relu')(x)
y = tf.keras.layers.Dense(10, activation='softmax')(net)
acc_value = tf.reduce_mean(
    tf.keras.metrics.categorical_accuracy(y_, y))

loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_, y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


# ### 2. 模型训练。

# In[2]:


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(3000):
        xs, ys = mnist_data.train.next_batch(100)
        _, loss_value = sess.run([train_step, loss], feed_dict={x: xs, y_: ys})
        if i % 1000 == 0:
            print("After %d training step(s), loss on training batch is "
                  "%g." % (i, loss_value))

    print acc_value.eval(feed_dict={x: mnist_data.test.images,
                                    y_: mnist_data.test.labels})

