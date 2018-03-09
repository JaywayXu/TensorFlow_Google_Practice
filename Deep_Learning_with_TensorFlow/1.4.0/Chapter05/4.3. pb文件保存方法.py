import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)  # 初始化所有变量
    # 导出当前计算图的GraphDef部分，只需要这一部分就可以完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()
    # 将需要保存的add节点名称传入参数中，表示将所需的变量转化为常量保存下来。
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
    # 将导出的模型存入文件中
    with tf.gfile.GFile("Saved_model/combined_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())

# #### 2. 加载pb文件。

from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = "Saved_model/combined_model.pb"
    # 读取保存的模型文件，并将其解析成对应的GraphDef Protocol Buffer
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # 将graph_def中保存的图加载到当前图中，其中保存的时候保存的是计算节点的名称，为add
    # 但是读取时使用的是张量的名称所以是add:0
    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    print(sess.run(result))
# Converted 2 variables to const ops.
# [array([3.], dtype=float32)]