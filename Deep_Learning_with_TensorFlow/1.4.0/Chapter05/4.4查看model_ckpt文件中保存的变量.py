import tensorflow as tf

# Tensorflow 提供了tf.train.NewCheckpointReader类来查看model.ckpt文件中保存的变量信息
reader = tf.train.NewCheckpointReader('Saved_model/model.ckpt')

# 获取所有变量列表，这是一个从变量名到变量维度的字典
all_variables = reader.get_variable_to_shape_map()
for variable_name in all_variables:
    # variable_name 为变量的名称，all_variable[variable_name]为变量的维度
    print(variable_name, all_variables[variable_name])
# 获取名称为V1的变量的取值
print('Value for Variable_1 is ', reader.get_tensor("Variable_1"))

# Variable_1 [1]  变量Variable_1的维度是[1]
# Variable [1]  变量Variable的维度是[1]
# Value for Variable_1 is  [-0.8113182]  变量Variable_1的取值是[-0.8113182]
