# #### 假设我们要最小化函数  $y=x^2$, 选择初始点   $x_0=5$
# #### 1. 学习率为1的时候，x在5和-5之间震荡。
import tensorflow as tf
TRAINING_STEPS = 10
LEARNING_RATE = 1
x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)

train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        sess.run(train_op)
        x_value = sess.run(x)
        print("After %s iteration(s): x%s is %f."% (i+1, i+1, x_value))

# #### 2. 学习率为0.001的时候，下降速度过慢，在901轮时才收敛到0.823355。

TRAINING_STEPS = 1000
LEARNING_RATE = 0.001
x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)

train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        sess.run(train_op)
        if i % 100 == 0: 
            x_value = sess.run(x)
            print("After %s iteration(s): x%s is %f."% (i+1, i+1, x_value))
# #### 3. 使用指数衰减的学习率，在迭代初期得到较高的下降速度，可以在较小的训练轮数下取得不错的收敛程度。

TRAINING_STEPS = 100
global_step = tf.Variable(0)
LEARNING_RATE = tf.train.exponential_decay(0.1, global_step, 1, 0.96, staircase=True)

x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
y = tf.square(x)
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS):
        sess.run(train_op)
        if i % 10 == 0:
            LEARNING_RATE_value = sess.run(LEARNING_RATE)
            x_value = sess.run(x)
            print("After %s iteration(s): x%s is %f, learning rate is %f."% (i+1, i+1, x_value, LEARNING_RATE_value))


# After 1 iteration(s): x1 is -5.000000.
# After 2 iteration(s): x2 is 5.000000.
# After 3 iteration(s): x3 is -5.000000.
# After 4 iteration(s): x4 is 5.000000.
# After 5 iteration(s): x5 is -5.000000.
# After 6 iteration(s): x6 is 5.000000.
# After 7 iteration(s): x7 is -5.000000.
# After 8 iteration(s): x8 is 5.000000.
# After 9 iteration(s): x9 is -5.000000.
# After 10 iteration(s): x10 is 5.000000.
# After 1 iteration(s): x1 is 4.990000.
# After 101 iteration(s): x101 is 4.084646.
# After 201 iteration(s): x201 is 3.343555.
# After 301 iteration(s): x301 is 2.736923.
# After 401 iteration(s): x401 is 2.240355.
# After 501 iteration(s): x501 is 1.833880.
# After 601 iteration(s): x601 is 1.501153.
# After 701 iteration(s): x701 is 1.228794.
# After 801 iteration(s): x801 is 1.005850.
# After 901 iteration(s): x901 is 0.823355.
# After 1 iteration(s): x1 is 4.000000, learning rate is 0.096000.
# After 11 iteration(s): x11 is 0.690561, learning rate is 0.063824.
# After 21 iteration(s): x21 is 0.222583, learning rate is 0.042432.
# After 31 iteration(s): x31 is 0.106405, learning rate is 0.028210.
# After 41 iteration(s): x41 is 0.065548, learning rate is 0.018755.
# After 51 iteration(s): x51 is 0.047625, learning rate is 0.012469.
# After 61 iteration(s): x61 is 0.038558, learning rate is 0.008290.
# After 71 iteration(s): x71 is 0.033523, learning rate is 0.005511.
# After 81 iteration(s): x81 is 0.030553, learning rate is 0.003664.
# After 91 iteration(s): x91 is 0.028727, learning rate is 0.002436.