import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

# #### 1. 定义神经网络的参数。
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 3000
MOVING_AVERAGE_DECAY = 0.99


# #### 2. 定义训练的过程并保存TensorBoard的log文件。
def train(mnist):
    #  输入数据的命名空间。
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 处理滑动平均的命名空间。
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())  # 对可训练变量集合使用滑动平均

    # 计算损失函数的命名空间。
    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        # 将交叉熵加上权值的正则化
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 定义学习率、优化方法及每一轮执行训练的操作的命名空间。
    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY,
            staircase=True)

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')
    # 将结果记录进log文件夹中
    writer = tf.summary.FileWriter("log", tf.get_default_graph())

    # 训练模型。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            if i%1000 == 0:
                # 配置运行时需要记录的信息。
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # 运行时记录运行信息的proto。
                run_metadata = tf.RunMetadata()
                # 将配置信息和记录运行信息的proto传入运行的过程，从而记录运行时每一个节点的时间空间开销信息
                _, loss_value, step = sess.run(
                    [train_op, loss, global_step], feed_dict={x: xs, y_: ys},
                    options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata=run_metadata, tag=("tag%d"%i), global_step=i)
                print("After %d training step(s), loss on training batch is %g."%(step, loss_value))
            else:
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

    writer.close()


# #### 3. 主函数。
def main(argv=None):
    mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()
# (C:\Users\cloud\AppData\Local\conda\conda\envs\py35) D:\CODE\Git\TensorFlow_Google_Practice>tensorboard --logdir=D:\CODE\Git\TensorFlow_Google_Practice\Deep_Learning_with_TensorFlow\1.4.0\Chapter11\log
