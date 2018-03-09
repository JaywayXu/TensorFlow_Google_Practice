import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train

# #### 1. 每10秒加载一次最新的模型
# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        # 直接通过调用封装好的函数来计算前向传播的结果，因为测试时不关注正则化损失的值所以这里用于计算正则化损失的函数被设置为None
        y = mnist_inference.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        # 如果需要离线预测未知数据的类别，只需要将计算正确率的部分改为答案的输出即可。
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过获取变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用滑动平均的函数来获取平均值
        # 这样可以完全共用mnist_inference.py重定义的前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存是迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g"%(global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)
            # 每次运行都是读取最新保存的模型，并在MNIST验证数据集上计算模型的正确率
            # 每隔EVAL_INTERVAL_SECS秒来调用一侧计算正确率的过程以检验训练过程中的正确率变化


# ###  主程序
def main(argv=None):
    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    main()

# After 29001 training step(s), validation accuracy = 0.9854
# 这个程序的本意是在train的同时也运行test的程序的每10秒就会调用一次测评函数。
