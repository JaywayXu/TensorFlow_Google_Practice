import tensorflow as tf
# 不同的命名空间
with tf.variable_scope("foo"):
    # 在命名空间foo下获取变量"bar",于是得到的变量名称为"foo/bar"
    a = tf.get_variable("bar", [1])
    print(a.name)
    # foo/bar:0
with tf.variable_scope("bar"):
    # 在命名空间bar下获取变量"bar",于是得到的变量名称为"bar/bar".此时变量在"bar/bar"和变量"foo/bar"并不冲突，于是可以正常运行
    b = tf.get_variable("bar", [1])
    print(b.name)
    # bar/bar:0
#  tf.Variable和tf.get_variable的区别。

with tf.name_scope("a"):
    # 使用tf.Variable函数生成变量时会受到tf.name_scope影响，于是这个变量的名称为"a/Variable"
    a = tf.Variable([1])
    print(a.name)
    # a/Variable: 0

    # tf.get_variable函数不受头tf.name_scope函数的影响，于是变量并不在a这个命名空间中
    a = tf.get_variable("b", [1])
    print(a.name)
    # b:0

# with tf.name_scope("b"):
    # 因为tf.get_variable不受tf.name_scope影响,所以这里将试图获取名称为"a"的变量。然而这个变量已经被声明了,于是这里会报重复声明的错误。
    # tf.get_variable("b",[1])
#  TensorBoard可以根据命名空间来整理可视化效果图上的节点。

with tf.name_scope("input1"):
    input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
with tf.name_scope("input2"):
    input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1, input2], name="add")

writer = tf.summary.FileWriter("log/simple_example.log", tf.get_default_graph())
writer.close()
# (C:\Users\cloud\AppData\Local\conda\conda\envs\py35) D:\CODE\Git\TensorFlow_Google_Practice>
# tensorboard --logdir=D:\CODE\Git\TensorFlow_Google_Practice\Deep_Learning_with_TensorFlow\1.4.0\Chapter11\log\simple_example.log
