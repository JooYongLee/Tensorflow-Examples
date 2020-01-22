import tensorflow as tf
import numpy as np

def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
        input_tensor, output_tensors = return_elements[0], return_elements[1:]

    return input_tensor, output_tensors

pb_file_path = "test_freee.pb"
graph = tf.Graph()
input_name_list = ["inputs:0", "test_output:0"
                   ]

input_tensor, output_tensors = read_pb_return_tensors(graph, pb_file_path,
                                                      input_name_list)

print("input---------->")
print(input_tensor)
print("output---------->")
print(output_tensors)

x = np.random.randn(1, 100, 200, 3)
sess = tf.Session(graph=graph)
out = sess.run(output_tensors[0],
         feed_dict={
             input_tensor:x
         })
print(out.shape)