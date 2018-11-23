import tensorflow as tf
from tensorflow.contrib import layers
import cnfig


l2_regularize = layers.l2_regularizer(scale=0.1)

class NerveTrackNet(object):
    def __init__(self,inputs, scope=''):
        with tf.name_scope("mdnets"):
            with tf.variable_scope(scope+"conv1"):
                nets = tf.layers.conv2d(inputs,
                                        filters=96,
                                        kernel_size=7,
                                        strides=2,
                                        activation=tf.nn.relu,
                                        use_bias=True,
                                        padding='valid',
                                        kernel_regularizer=l2_regularize,
                                        kernel_initializer= tf.truncated_normal_initializer
                                        )
                self.conv1 = nets

                nets = tf.layers.max_pooling2d(nets,3,2,padding='valid')

            with tf.variable_scope(scope+"conv2"):
                nets = tf.layers.conv2d(nets,
                                        filters=256,
                                        kernel_size=5,
                                        strides=2,
                                        activation=tf.nn.relu,
                                        use_bias=True,
                                        padding='valid',
                                        kernel_regularizer=l2_regularize,
                                        kernel_initializer= tf.truncated_normal_initializer)

                self.conv2 = nets

                nets = tf.layers.max_pooling2d(nets, 3, 2, padding='valid')




            with tf.variable_scope(scope+"conv3"):
                nets = tf.layers.conv2d(nets,
                                        filters=512,
                                        kernel_size=3,
                                        strides=1,
                                        activation=tf.nn.relu,
                                        use_bias=True,
                                        padding='valid',
                                        kernel_regularizer=l2_regularize,
                                        kernel_initializer= tf.truncated_normal_initializer)

                self.conv3 = nets
            with tf.variable_scope(scope+"fc4"):

                nets = tf.reshape(nets,shape=[-1, 512*3*3])
                nets = tf.layers.dense(nets,
                                       512,
                                       activation=tf.nn.relu,
                                       use_bias=True,
                                       kernel_regularizer=l2_regularize,
                                       kernel_initializer= tf.truncated_normal_initializer
                                       )

                self.fc4 = nets
            with tf.variable_scope(scope+"fc5"):
                nets = tf.layers.dense(nets,
                                       512,
                                       activation=tf.nn.relu,
                                       use_bias=True,
                                       kernel_regularizer=l2_regularize,
                                       kernel_initializer=tf.truncated_normal_initializer
                                       )
                self.fc5 = nets
            with tf.variable_scope(scope+"fc6"):
                nets = tf.layers.dense(nets,
                                       cnfig.NUM_CALSS,
                                       activation=None,
                                       use_bias=True,
                                       kernel_regularizer=l2_regularize,
                                       kernel_initializer=tf.truncated_normal_initializer
                                       )

                self.fc6 = nets
                self.softmax_fc6 = tf.nn.softmax(self.fc6)



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


def model_cnfigure_test() :
    """
    Constructs box collection.
    Args:
      boxes: a tensor of shape [N, 4] representing box corners
    Raises:
      ValueError: if invalid dimensions for bbox data or if bbox data is not in
          float32 format.

    """
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, cnfig.IMAGE_HEIGHT, cnfig.IMAGE_WIDTH, 1],name='nerve_track_inputs')
    labels = tf.placeholder(dtype=tf.float32, shape=[None, cnfig.NUM_CALSS],name='labels')
    nervenet = NerveTrackNet(inputs)
    # print_tensor()

    conv_vars = []
    fc_vars = []
    for vars in tf.trainable_variables():
        if 'conv' in vars.name:
            conv_vars.append(vars)
    for vars in tf.trainable_variables():
        if 'fc' in vars.name:
            fc_vars.append(vars)
    print("--------convolutional neural network variables--------")
    # print(conv_vars)
    [print(x) for x in conv_vars]
    print("--------fully connected neural network variables--------")
    [print(x) for x in fc_vars]
    print("---------output layer names--------------------")
    print(nervenet.fc6)


if __name__=="__main__":
    model_cnfigure_test()
    # lossfunc()
    # model_cnfigure_test()
    pass



