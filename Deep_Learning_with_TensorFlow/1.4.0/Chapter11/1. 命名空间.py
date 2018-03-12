import tensorflow as tf

with tf.variable_scope("foo"):
    a = tf.get_variable("bar", [1])
    print(a.name)
    # foo/bar:0
with tf.variable_scope("bar"):
    b = tf.get_variable("bar", [1])
    print(b.name)
    # bar/bar:0
# #### 2. tf.Variable和tf.get_variable的区别。

with tf.name_scope("a"):
    a = tf.Variable([1])
    print(a.name)
    # a/Variable: 0
    a = tf.get_variable("b", [1])
    print(a.name)
    # b:0
# #### 3. TensorBoard可以根据命名空间来整理可视化效果图上的节点。

with tf.name_scope("input1"):
    input1 = tf.constant([1.0, 2.0, 3.0], name="input2")
with tf.name_scope("input2"):
    input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1, input2], name="add")

writer = tf.summary.FileWriter("log/simple_example.log", tf.get_default_graph())
writer.close()
# (C:\Users\cloud\AppData\Local\conda\conda\envs\py35) D:\CODE\Git\TensorFlow_Google_Practice>tensorboard --logdir=D:\CODE\Git\TensorFlow_Google_Practice\Deep_Learning_with_TensorFlow\1.4.0\Chapter11\log
