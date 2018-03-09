import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
# v1= tf.Variable(0, dtype=tf.float32, name="v1")
#  这里如果新生成一个变量名为v1的变量使用ema.variables_to_restore()就会报错


ema = tf.train.ExponentialMovingAverage(0.99)
print(ema.variables_to_restore())
# {'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}

# saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
saver = tf.train.Saver(ema.variables_to_restore())
with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model2.ckpt")
    print(sess.run(v))
    # print(sess.run(v1))
# 0.099999905
