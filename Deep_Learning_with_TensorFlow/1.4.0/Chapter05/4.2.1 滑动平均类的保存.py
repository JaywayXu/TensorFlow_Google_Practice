import tensorflow as tf

# #### 1. 使用滑动平均。
# 由于没有使用tf.global_variables_initializer()函数
# 所以此处我们只是输出了变量的名称没有输出变量的值
# 如果需要得到变量的值要通过init()和sess.run()才能得到

v = tf.Variable(0, dtype=tf.float32, name="v")
for variables in tf.global_variables():
    print('The name of variables is:', variables.name)
    # The name of variables is: v:0
ema = tf.train.ExponentialMovingAverage(0.99)  # 滑动平均操作的声明
maintain_averages_op = ema.apply(tf.global_variables())
for variables in tf.global_variables():
    print('use ExponentialMovingAverage ,the name of variabes is:', variables.name)
    # use ExponentialMovingAverage ,the name of variabes is: v:0
    # use ExponentialMovingAverage ,the name of variabes is: v/ExponentialMovingAverage:0
# #### 2. 保存滑动平均模型。

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    sess.run(tf.assign(v, 10))  # 对变量V进行赋值
    sess.run(maintain_averages_op)  # 对变量V进行滑动平均操作
    # 保存的时候会将v:0  v/ExponentialMovingAverage:0这两个变量都存下来。
    saver.save(sess, "Saved_model/model2.ckpt")
    print('the value of v and ema.average(v)', sess.run([v, ema.average(v)]))
    # the value of v and ema.average(v) [10.0, 0.099999905]
# #### 3. 加载滑动平均模型。

v = tf.Variable(0, dtype=tf.float32, name="v")

# 通过变量重命名将原来变量v的滑动平均值直接赋值给v。
saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
# 将日志文件中变量名称为v/ExponentialMovingAverage的值赋给现在名称为v的变量
with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model2.ckpt")
    print('the value of v/ExponentialMovingAverage from  Saved_model/model2.ckpt', sess.run(v))
# 0.099999905
