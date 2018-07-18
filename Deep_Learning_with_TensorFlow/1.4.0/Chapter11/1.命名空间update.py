import tensorflow as tf
with tf.name_scope("input1"):
    input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
with tf.name_scope("input2"):
    input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1, input2], name="add")

writer = tf.summary.FileWriter("log/simple_example.log", tf.get_default_graph())
writer.close()
# (C:\Users\cloud\AppData\Local\conda\conda\envs\py35) D:\CODE\Git\TensorFlow_Google_Practice>
# tensorboard --logdir=D:\CODE\Git\TensorFlow_Google_Practice\Deep_Learning_with_TensorFlow\1.4.0\Chapter11\log\simple_example.log
