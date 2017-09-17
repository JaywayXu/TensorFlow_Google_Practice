import tensorflow as tf

"""只有顶层的命名空间中的节点才会被显示,tf.name_scope和tf.variable_scope函数都可以管理变量的命名空间"""
with tf.variable_scope("foo"):
    # 在命名空间foo下获取变量"bar",于是得到的变量名称为"foo/bar"
    a = tf.get_variable(name="bar", shape=[1])
    print(a.name)

with tf.variable_scope("bar"):
    b = tf.get_variable(name="bar", shape=[1])
    # 在命名空间bar下获取变量"bar",于是得到的变量名称为"bar/bar"
    # 此时的两个变量并不冲突.
    print(b.name)
    # foo/bar: 0
    # bar/bar: 0

"""tf.Variable和tf.get_variable的区别
tf.variable
def __init__(self,
               initial_value=None,
               trainable=True,
               collections=None,
               validate_shape=True,
               caching_device=None,
               name=None,
               variable_def=None,
               dtype=None,
               expected_shape=None,
               import_scope=None):

def get_variable(name,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=True,
                 collections=None,
                 caching_device=None,
                 partitioner=None,
                 validate_shape=True,
                 use_resource=None,
                 custom_getter=None):
"""

with tf.name_scope("a"):
    a = tf.Variable(1, name="variable_0")
    print(a.name)
    #  使用tf.Variable函数生成的变量会受tf.name_scope影响,于是这个变量的名称为"a/variable_0:0"

    a = tf.get_variable(name="b", shape=[1])  # 获取名称为b的变量
    print(a.name)
    # 使用tf.get_variable函数生成的变量不会受tf.name_scope的影响.
    # a/variable_0: 0
    # b: 0  即这个变量不会在a的命名范围内
with tf.name_scope("input1"):
    input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
with tf.name_scope("input2"):
    input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1, input2], name="add")

writer = tf.summary.FileWriter("./log_01", tf.get_default_graph())
writer.close()

# tensorboard --logdir=F:\Git\TensorFlow_Google_Practice\Deep_Learning_with_TensorFlow\1.0.0\Chapter09\log_01