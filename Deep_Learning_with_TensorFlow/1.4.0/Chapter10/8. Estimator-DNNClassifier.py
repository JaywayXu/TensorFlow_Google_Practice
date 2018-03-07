
# coding: utf-8

# ### 1. 模型定义。

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.INFO)

mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=False)

# 定义模型的输入。
feature_columns = [tf.feature_column.numeric_column("image", shape=[784])]

# 通过DNNClassifier定义模型。
estimator = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                       hidden_units=[500],
                                       n_classes=10,
                                       optimizer=tf.train.AdamOptimizer(),
                                       model_dir="log")


# ### 2. 训练模型。

# In[2]:


train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"image": mnist.train.images},
      y=mnist.train.labels.astype(np.int32),
      num_epochs=None,
      batch_size=128,
      shuffle=True)

estimator.train(input_fn=train_input_fn, steps=10000)


# ### 3. 测试模型。

# In[3]:


test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"image": mnist.test.images},
      y=mnist.test.labels.astype(np.int32),
      num_epochs=1,
      batch_size=128,
      shuffle=False)

test_results = estimator.evaluate(input_fn=test_input_fn)
accuracy_score = test_results["accuracy"]
print("\nTest accuracy: %g %%" % (accuracy_score*100))

print test_results

