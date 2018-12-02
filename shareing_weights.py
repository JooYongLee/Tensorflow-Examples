import tensorflow as tf
import numpy as np
def fc(X,scope):
    with tf.variable_scope(scope, "fc1", reuse=tf.AUTO_REUSE):
        w = tf.get_variable(name="w1",shape=[10, 10],dtype=tf.float32)
        netmodels = tf.matmul(X,w)
        return netmodels

X = tf.placeholder(dtype=tf.float32, shape=[None, 10])
X1 = tf.placeholder(dtype=tf.float32, shape=[None, 10])
nets = fc(X,"A")
nets = fc(nets,"B")
nets2 = fc(X1,"B")
print(tf.trainable_variables())
x = np.random.randn(10,10)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    n1,n2 = sess.run([nets, nets2],feed_dict={X:x, X1:x})
    print(n1.shape)
    print(n2.shape)



