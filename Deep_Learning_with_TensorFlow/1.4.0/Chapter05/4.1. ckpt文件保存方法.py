import tensorflow as tf

# #### 1. 保存计算两个变量和的模型。

v1 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))
v2 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))
result = v1 + v2

init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "Saved_model/model.ckpt")

# #### 2. 加载保存了两个变量和的模型。

with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model.ckpt")
    print(sess.run(result))
    # [-1.6226364]
# #### 3. 直接加载持久化的图。因为之前没有导出v3，所以这里会报错。

saver = tf.train.import_meta_graph("Saved_model/model.ckpt.meta")
v3 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))

with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model.ckpt")
    print(sess.run(v1))
    print(sess.run(v2))
    # [-0.8113182]
    # [-0.8113182]
    # print(sess.run(v3))  # 此语句会报如下错误，现在我们把它注释掉
    # 注意在这个sess的管理器中没有对v3进行变量值的初始化
    # 即没有使用tf.tf.global_variables_initializer()语句，我们的v1和v2都是从saved_model/model.ckpt
    # 文件中加载的值，

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")
saver = tf.train.Saver({"v1": v1, "v2": v2})
# Saver函数默认的是保存所有的变量值，我们现在指定只保存v1和v2这两个变量的值
# 在这个程序中，对变量v1和v2的名称进行了修改，如果直接通过tf.train.Saver默认的构造函数来加载保存的模型，
# 那么程序会报找不到变量的错误，因为保存时候变量的名称和加载时变量的名称不一致。
# 为了解决这个问题，我们可以通过字典的方式，把变量联系起来。
# 其中"v1"表示文件中保存的变量名，v1表示当前程序中使用的变量名。
