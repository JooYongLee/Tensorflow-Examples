import tensorflow as tf



def freeze_graph(sess, output_file, output_node_names):

    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        output_node_names,
    )

    with tf.gfile.GFile(output_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("=> {} ops written to {}.".format(len(output_graph_def.node), output_file))



def build(x):

    # x = net_inputs
    x = tf.layers.conv2d(x, 8, 3, strides=(2, 2), name="conv1")
    print(x.shape)
    x = tf.layers.conv2d(x, 16, 3, strides=(2, 2), name="conv2")
    x = tf.layers.conv2d(x, 32, 3, strides=(2, 2), name="conv3")
    return x

with tf.Graph().as_default() as graph:
    sess = tf.Session(graph=graph)

    net_inputs = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3], name="inputs")

    x = build(net_inputs)
    # sess.run(tf.initialize_variables())
    sess.run(tf.global_variables_initializer())



    out_node_name = "test_output"
    out_tensor = tf.identity(x, name=out_node_name)
    output_node_names = [out_node_name]
    print("input-->", net_inputs.name)
    print("output-------->", output_node_names)
    nodes_list = [n.name for n in sess.graph.as_graph_def().node]
    # for n in nodes_list:
    #     print(n)
    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


    # print("train variable----->", vars)
    # saver = tf.train.Saver(var_list=vars)

    freeze_graph(sess, "test_freee.pb", output_node_names)


