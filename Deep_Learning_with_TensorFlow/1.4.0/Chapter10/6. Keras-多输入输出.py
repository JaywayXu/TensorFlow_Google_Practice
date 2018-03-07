
# coding: utf-8

# ### 1. 数据预处理。

# In[1]:


import keras
from tflearn.layers.core import fully_connected
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model

num_classes = 10
img_rows, img_cols = 28, 28
 
# 通过Keras封装好的API加载MNIST数据。
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX.reshape(trainX.shape[0], img_rows * img_cols)
testX = testX.reshape(testX.shape[0], img_rows * img_cols)

trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255.0
testX /= 255.0

trainY = keras.utils.to_categorical(trainY, num_classes)
testY = keras.utils.to_categorical(testY, num_classes)


# ### 2. 定义模型。

# In[2]:


# 定义两个输入。
input1 = Input(shape=(784,), name = "input1")
input2 = Input(shape=(10,), name = "input2")

# 定义第一个输出。
x = Dense(1, activation='relu')(input1)
output1 = Dense(10, activation='softmax', name = "output1")(x)

# 定义第二个输出。
y = keras.layers.concatenate([x, input2])
output2 = Dense(10, activation='softmax', name = "output2")(y)

model = Model(inputs=[input1, input2], outputs=[output1, output2])

# 定义损失函数、优化函数和评测方法。
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              loss_weights = [1, 0.1],
              metrics=['accuracy'])


# ### 3. 模型训练。

# In[3]:


model.fit([trainX, trainY], [trainY, trainY],
          batch_size=128,
          epochs=20,
          validation_data=([testX, testY], [testY, testY]))

