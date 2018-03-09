import tensorflow as tf

# 定义神经网络结构相关参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


# 设置权值函数
# 在训练时会创建这些变量，在测试时会通过保存的模型加载这些变量的取值
# 因为可以在变量加载时将滑动平均变量均值重命名，所以这个函数可以直接通过同样的名字在训练时使用变量本身
# 而在测试时使用变量的滑动平均值，在这个函数中也会将变量的正则化损失加入损失集合

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    # 如果使用正则化方法会将该张量加入一个名为'losses'的集合
    if regularizer != None: tf.add_to_collection('losses', regularizer(weights))
    return weights


# 定义神经网络前向传播过程
def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2
