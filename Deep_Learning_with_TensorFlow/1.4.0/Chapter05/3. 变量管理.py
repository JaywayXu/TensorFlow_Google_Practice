import tensorflow as tf

# #### 1. 在上下文管理器“foo”中创建变量“v”。
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))

# with tf.variable_scope("foo"):
# v = tf.get_variable("v", [1])

with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
print(v == v1)
# True

# with tf.variable_scope("bar", reuse=True):
# v = tf.get_variable("v", [1])


# #### 2. 嵌套上下文管理器中的reuse参数的使用。

with tf.variable_scope("root"):
    print(tf.get_variable_scope().reuse)
    # False 最外层没有明确指出时reuse为false,表示创建新的变量
    with tf.variable_scope("foo", reuse=True):
        print(tf.get_variable_scope().reuse)
        # True
        with tf.variable_scope("bar"):
            print(tf.get_variable_scope().reuse)
        # True 嵌套类型默认和外层保持一致
    print(tf.get_variable_scope().reuse)
    # False 此处表明最外层的variable_scope属性
# #### 3. 通过variable_scope来管理变量。
v1 = tf.get_variable("v", [1])
print(v1.name)
# v:0 V表示变量名称，0表示这个变量是生成变量这个运算的第一个结果
with tf.variable_scope("foo", reuse=True):
    v2 = tf.get_variable("v", [1])
print(v2.name)
# foo/v:0

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v", [1])
        print(v3.name)
# foo/bar/v: 0
# 命名空间可以嵌套，同时变量名称也会加入所有命名空间的名称作为前缀
v4 = tf.get_variable("v1", [1])
print(v4.name)
# v1:0
# #### 4. 我们可以通过变量的名称来获取变量。

with tf.variable_scope("", reuse=True):
    v5 = tf.get_variable("foo/bar/v", [1])
    print(v5 == v3)
    # True

    v6 = tf.get_variable("v1", [1])
    print(v6 == v4)
    # True
