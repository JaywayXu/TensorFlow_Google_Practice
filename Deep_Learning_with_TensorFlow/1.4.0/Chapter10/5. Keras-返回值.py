
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
 
# 将图像像素转化为0到1之间的实数。
trainX = trainX.astype('float32')
testX = testX.astype('float32')
trainX /= 255.0
testX /= 255.0
 
# 将标准答案转化为需要的格式（one-hot编码）。
trainY = keras.utils.to_categorical(trainY, num_classes)
testY = keras.utils.to_categorical(testY, num_classes)


# ### 2. 通过返回值的方式定义模型。

# In[2]:


inputs = Input(shape=(784,))

x = Dense(500, activation='relu')(inputs)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])


# ### 3. 训练模型。

# In[3]:


model.fit(trainX, trainY,
          batch_size=32,
          epochs=10,
          validation_data=(testX, testY))

