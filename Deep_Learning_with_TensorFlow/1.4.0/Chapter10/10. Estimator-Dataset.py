
# coding: utf-8

# ### 1. 训练自定义模型。

# In[1]:


import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
        return {"x": parsed_line[:-1]}, parsed_line[-1:]

    dataset = (tf.contrib.data.TextLineDataset(file_path)  
               .skip(1)  
               .map(decode_csv))  
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count) 
    dataset = dataset.batch(32) 
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,  
    hidden_units=[10, 10],  
    n_classes=3)

classifier.train(
    input_fn=lambda: my_input_fn("../../datasets/iris_training.csv", True, 100))


# ### 2. 测试模型。

# In[2]:


test_results = classifier.evaluate(
    input_fn=lambda: my_input_fn("../../datasets/iris_test.csv", False, 1))
print("\nTest accuracy: %g %%" % (test_results["accuracy"]*100))

