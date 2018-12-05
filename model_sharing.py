import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import re
import time

l2_regularize = layers.l2_regularizer(scale=0.3)
tf_initalizer = None

class NerveTrackNet(object):
    def __init__(self,inputs, feature, scope=''):
        self.num_class = 2

        self.buildConvNet(inputs, scope)

        self.predit = self.buildFCNet(self.conv3, "direct")
        self.predit_from_feat = self.buildFCNet(feature, "feat")

        self._collection_variables()



    def get_direct_fclist(self):
        return self.predit
    def get_feat_fcllist(self):
        return self.predit_from_feat

    def buildConvNet(self, inputs, scope):
        with tf.name_scope("mdnets"):
            with tf.variable_scope(scope + "conv1"):
                nets = tf.layers.conv2d(inputs,
                                        filters=96,
                                        kernel_size=7,
                                        strides=2,
                                        activation=tf.nn.relu,
                                        use_bias=True,
                                        padding='valid',
                                        kernel_regularizer=l2_regularize,
                                        kernel_initializer=tf_initalizer
                                        )
                self.conv1 = nets
                _activation_summary(nets)
                nets = tf.layers.max_pooling2d(nets, 3, 2, padding='valid')

            with tf.variable_scope(scope + "conv2"):
                nets = tf.layers.conv2d(nets,
                                        filters=256,
                                        kernel_size=5,
                                        strides=2,
                                        activation=tf.nn.relu,
                                        use_bias=True,
                                        padding='valid',
                                        kernel_regularizer=l2_regularize,
                                        kernel_initializer=tf_initalizer)
                _activation_summary(nets)

                self.conv2 = nets

                nets = tf.layers.max_pooling2d(nets, 3, 2, padding='valid')

            with tf.variable_scope(scope + "conv3"):
                nets = tf.layers.conv2d(nets,
                                        filters=512,
                                        kernel_size=3,
                                        strides=1,
                                        activation=tf.nn.relu,
                                        use_bias=True,
                                        padding='valid',
                                        kernel_regularizer=l2_regularize,
                                        kernel_initializer=tf_initalizer
                                        )

                self.conv3 = nets
    def buildFCNet(self, inputs, scope):
        with tf.name_scope(scope):
            with tf.variable_scope("FCNet",reuse=tf.AUTO_REUSE):
                with tf.variable_scope("fc4"):
                    nets = tf.reshape(inputs, shape=[-1, 512 * 3 * 3])
                    nets = tf.layers.dense(nets,
                                           512,
                                           activation=tf.nn.relu,
                                           use_bias=False,
                                           kernel_regularizer=l2_regularize,
                                           kernel_initializer=tf_initalizer
                                           )

                    fc4 = nets
                with tf.variable_scope("fc5"):
                    nets = tf.layers.dense(nets,
                                           512,
                                           activation=tf.nn.relu,
                                           use_bias=False,
                                           kernel_regularizer=l2_regularize,
                                           kernel_initializer=tf_initalizer
                                           )
                    fc5 = nets
                with tf.variable_scope("fc6"):
                    nets = tf.layers.dense(nets,
                                           self.num_class,
                                           activation=None,
                                           use_bias=False,
                                           kernel_regularizer=l2_regularize,
                                           kernel_initializer=tf_initalizer
                                           )

                # self.fc6 = nets
                # self.softmax_fc6 = tf.nn.softmax(self.fc6)
                fc6 = nets
                fc6softmax = tf.nn.softmax(fc6)
                return {"fc6":fc6, "fc6softmax":fc6softmax}

    def _collection_variables(self):
        scope_list = ["conv1", "conv2", "conv3"]
        conv_vars = []
        for scope in scope_list:
            conv_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        self.convNet_vars = conv_vars

        scope_list = ["fc4", "fc5", "fc6"]
        fc_vars = []
        for scope in scope_list:
            fc_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        self.FC_vars = fc_vars

    def get_prediction_logits(self):
        return self.fc6
    def get_softmax_logtis(self):
        return self.softmax_fc6
    def get_feature_list(self):
        return [self.conv1, self.conv2, self.conv3]
    def get_convNetWeights(self):
        return self.convNet_vars
    def get_convNet_layer(self):
        return self.conv3
    def get_FCWeights(self):
        return self.FC_vars


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
    x: Tensor
    Returns:
    nothing
    """
    TOWER_NAME = 'tower'
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

def lossfunc(model_outputs, label):
    """
    :param model_outputs:model class nerver detection
    :param label: labels tensor
    :return:
    """

    loss = tf.nn.softmax_cross_entropy_with_logits( logtis = model_outputs, labels = label)
    optimizer = tf.train.AdamOptimizer(learning_rate=cnfig.LEARNING_RATE ).minimize(loss)
    return optimizer, loss



def print_tensor():
    op = tf.get_default_graph().get_operations()
    for iop in op:
        print(iop.name, iop.values())

def print_trainable_var():
    """

    :return:
    """
    vars = tf.trainable_variables()
    for var in vars:
        print(var.name, var)

def loss_compare_time(sess, nervenet, inputs, labels, feat):
    feat_pred = nervenet.get_feat_fcllist()
    input_pred = nervenet.get_direct_fclist()

    loss_feat = tf.nn.softmax_cross_entropy_with_logits(logits=feat_pred["fc6"], labels=labels)
    loss_direct = tf.nn.softmax_cross_entropy_with_logits(logits=input_pred["fc6"], labels=labels)

    opt_feat = tf.train.AdamOptimizer(0.01).minimize(loss_feat)
    opt_direct = tf.train.AdamOptimizer(0.01).minimize(loss_direct)
    sess.run(tf.global_variables_initializer())

    batch = 10
    x = np.random.randn(batch,107,107,3)
    y = np.random.normal(1,0.01,[10,2])
    feat_x = np.random.randn(batch,3,3,512)

    iter = 100
    tact_feat = 0
    tact_dir = 0
    for cnt in range(iter):
        feat_start = time.time()
        _ = sess.run(opt_feat,feed_dict={labels:y,feat:feat_x})
        feat_end = time.time()

        dir_start = time.time()
        _ = sess.run(opt_direct, feed_dict={inputs: x, labels:y })
        dir_end = time.time()

        tact_feat += (feat_end - feat_start)
        tact_dir += (dir_end - dir_start)

        if cnt % 10 == 0 and cnt > 0:
            print("feat {}: direct{}".format(tact_feat/cnt, tact_dir/cnt))

    print("feat {} ,, direct {}".format(tact_feat/iter, tact_dir/iter))


def model_cnfigure_test():
    """
    Constructs box collection.
    Args:
      boxes: a tensor of shape [N, 4] representing box corners
    Raises:
      ValueError: if invalid dimensions for bbox data or if bbox data is not in
          float32 format.

    """
    tic=time.time()
    opts = {}
    opts['img_size'] = 107
    # opts['padding'] = 16
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, opts['img_size'], opts['img_size'], 3],name='nerve_track_inputs')
    labels = tf.placeholder(dtype=tf.float32, shape=[None, 2],name='labels')
    feat = tf.placeholder(dtype=tf.float32, shape=[None,3,3,512],name='feature')
    nervenet = NerveTrackNet(inputs,feat)
    # print_tensor()
    summary = tf.summary.FileWriter("./testgraph",tf.get_default_graph())
    sess = tf.Session()
    loss_compare_time(sess, nervenet, inputs, labels, feat)
    conv_vars = []
    fc_vars = []
    for vars in tf.trainable_variables():
        if 'conv' in vars.name:
            conv_vars.append(vars)
    for vars in tf.trainable_variables():
        if 'fc' in vars.name:
            fc_vars.append(vars)

    print(nervenet.get_direct_fclist())
    print(nervenet.get_feat_fcllist())
    print("--------convolutional neural network variables--------")
    # print(conv_vars)
    [print(x) for x in conv_vars]
    print("--------fully connected neural network variables--------")
    [print(x) for x in fc_vars]
    print("---------output layer names--------------------")
    toc=time.time()
    print("time {}".format(toc-tic))
    # print(nervenet.fc6)


if __name__=="__main__":
    model_cnfigure_test()
    # lossfunc()
    # model_cnfigure_test()
    pass



