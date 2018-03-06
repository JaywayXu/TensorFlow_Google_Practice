import tensorflow as tf
from numpy.random import RandomState

# #### 1. 定义神经网络的参数，输入和输出节点。

batch_size = 8
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# #### 2. 定义前向传播过程，损失函数及反向传播算法。

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                + (1 - y_)*tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# ####  3. 生成模拟数据集。

rdm = RandomState(1)
X = rdm.rand(128, 2)
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# #### 4. 创建一个会话来运行TensorFlow程序。

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 输出目前（未经训练）的参数取值。
    print('输出未经过训练的W1值\n', sess.run(w1))
    # 输出未经过训练的W1值
    # [[-0.8113182   1.4845988   0.06532937]
    #  [-2.4427042   0.0992484   0.5912243]]
    print('输出未经过训练的W2值\n', sess.run(w2))
    # 输出未经过训练的W2值
    # [[-0.8113182]
    #  [1.4845988]
    #  [0.06532937]]
    print("\n")

    # 训练模型。
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size)%128
        end = (i*batch_size)%128 + batch_size
        sess.run([train_step, y, y_], feed_dict={x: X[start:end], y_: Y[start:end]})
        if i%1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g"%(i, total_cross_entropy))

# After 0 training step(s), cross entropy on all data is 1.89805
# After 1000 training step(s), cross entropy on all data is 0.655075
# After 2000 training step(s), cross entropy on all data is 0.626172
# After 3000 training step(s), cross entropy on all data is 0.615096
# After 4000 training step(s), cross entropy on all data is 0.610309

    # 输出训练后的参数取值。
    print("\n")
    print('输出训练后的W1值\n', sess.run(w1))
    # 输出训练后的W1值
    # [[0.02476984  0.5694868   1.6921942]
    #  [-2.1977348 - 0.23668921  1.1143897]]
    print('输出训练后的W2值\n', sess.run(w2))
    # 输出训练后的W2值
    #  [[-0.45544702]
    #  [ 0.4911093 ]
    #  [-0.9811033 ]]
