
# coding: utf-8

# In[1]:


import tensorflow as tf


# #### 1. 在上下文管理器“foo”中创建变量“v”。

# In[2]:


with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))
                        
#with tf.variable_scope("foo"):
   # v = tf.get_variable("v", [1])
    
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
print v == v1

#with tf.variable_scope("bar", reuse=True):
   # v = tf.get_variable("v", [1])


# #### 2. 嵌套上下文管理器中的reuse参数的使用。

# In[3]:


with tf.variable_scope("root"):
    print tf.get_variable_scope().reuse
    
    with tf.variable_scope("foo", reuse=True):
        print tf.get_variable_scope().reuse
        
        with tf.variable_scope("bar"):
            print tf.get_variable_scope().reuse
            
    print tf.get_variable_scope().reuse


# #### 3. 通过variable_scope来管理变量。

# In[4]:


v1 = tf.get_variable("v", [1])
print v1.name

with tf.variable_scope("foo",reuse=True):
    v2 = tf.get_variable("v", [1])
print v2.name

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v", [1])
        print v3.name
        
v4 = tf.get_variable("v1", [1])
print v4.name


# #### 4. 我们可以通过变量的名称来获取变量。

# In[5]:


with tf.variable_scope("",reuse=True):
    v5 = tf.get_variable("foo/bar/v", [1])
    print v5 == v3
    v6 = tf.get_variable("v1", [1])     
    print v6 == v4

