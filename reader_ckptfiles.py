from tensorflow.python import pywrap_tensorflow

reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
# if all_tensors:
var_to_shape_map = reader.get_variable_to_shape_map()

