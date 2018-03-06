import tensorflow as tf

# 计算图
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", [1], initializer=tf.zeros_initializer())  # 设置初始值为0

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", [1], initializer=tf.ones_initializer())  # 设置初始值为1

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

# [ 0.]
# [ 1.]

# 张量
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b
print(result)

sess = tf.InteractiveSession()
print(result.eval())
sess.close()

# Tensor("add:0", shape=(2,), dtype=float32)
# [ 3.  5.]

# 3. 会话的使用
# 3.1 创建和关闭会话
# 创建一个会话。
sess = tf.Session()

# 使用会话得到之前计算的结果。
print(sess.run(result))

# 关闭会话使得本次运行中使用到的资源可以被释放。
sess.close()
# [ 3.  5.]
# 3.2 使用with statement 来创建会话
with tf.Session() as sess:
    print(sess.run(result))
# [3.  5.]
# 3.3 指定默认会话
sess = tf.Session()
with sess.as_default():
    print(result.eval())
# [3.  5.]
sess = tf.Session()

# 下面的两个命令有相同的功能。
print(sess.run(result))
print(result.eval(session=sess))
# [ 3.  5.]
#
# [ 3.  5.]
# 4. 使用tf.InteractiveSession构建会话
sess = tf.InteractiveSession()
print(result.eval())
sess.close()
# [ 3.  5.]
# 5. 通过ConfigProto配置会话
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
sess1 = tf.InteractiveSession(config=config)
sess2 = tf.Session(config=config)
