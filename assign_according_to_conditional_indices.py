import tensorflow as tf
import numpy as np
def test_assign_from_where_conditoins( ):
    # set input values
    x_mean = 10.0
    x_std = 5.0
    x = np.random.normal( x_mean, x_std, [5, 4] )
    # print( "value {}, shape :{}".format(x, x.shape) )
    X = tf.Variable(x , name='x', dtype = tf.int32 )
    # X = tf.constant(x, name='x', dtype=tf.int32)

    ############## conditions
    change_x = 0
    condi_X = tf.where(  X < 10, tf.ones_like( X ) * change_x, X) # change values according to conditional indices
    # condi_X = tf.where(X < 10) # get indices
    # condi_X = tf.where(X < 10, tf.ones_like(X) * change_x) # error case
    #condi_X = tf.where(X < 10, tf.zeros_like(X), X)

    ############# change value
    #X_assign = X.assign( condi_X )
    X.assign( condi_X )
    # X_assign = tf.assign( X, condi_X )

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run( init )
        # print( sess.run( condi_X ))

        print("before :\n {} \n After : \n{}".format(\
                            sess.run( X ),
                            sess.run( condi_X )
                            ))

    # print( tf.get_default_graph().as_graph_def().node)
def test_change_value_spare_tensor():
    # set input values
    x_mean = 10.0
    x_std = 5.0
    x = np.random.normal(x_mean, x_std, [5, 4])
    # print( "value {}, shape :{}".format(x, x.shape) )
    X = tf.Variable(x, name='x', dtype=tf.int32)

    ############## conditions
    change_x = -10
    condi_X = tf.where(X < 10)
    mask = X < 10
    # condi_X = tf.where(X < 10, tf.zeros_like(X), X)
    #
    # X_assign = tf.SparseTensor(
    #     indices= condi_X,
    #     values= tf.zeros_like( X ),
    #     dense_shape= X.shape
    # )
    # sparse_x_assign = tf.sparse_to_dense( X_assign, output_shape= X.shape)


    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run( init )

        # print( sess.run( sparse_x_assign) )


    pass
def test_change_tensor_by_boolean_mask():
    # set input values
    x_mean = 10.0
    x_std = 5.0
    x = np.random.normal(x_mean, x_std, [5, 4])
    # print( "value {}, shape :{}".format(x, x.shape) )
    X = tf.Variable(x, name='x', dtype=tf.int32)

    ############## conditions
    # get conditional, or mask
    mask =  X < 10
    cmask = X >= 10


    # change the value according to mask


    # change value tensor of indices using conditional mask and complement of conditional mask
    compute_mask = tf.cast(mask, dtype = tf.int32 )  * tf.constant(10, tf.int32)
    complement_mask = tf.cast(cmask, dtype=tf.int32) * X
    assign_mask = compute_mask + complement_mask


    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run( init )

        print('input value\n', sess.run(X))

        print('assign_mask\n', sess.run(assign_mask))

def test_boolean_mask_scatter_update():
    # set input values
    x_mean = 10.0
    x_std = 5.0
    x = np.random.normal(x_mean, x_std, [5, 4])
    # print( "value {}, shape :{}".format(x, x.shape) )
    X = tf.Variable(x, name='x', dtype=tf.int32)

    ############## conditions
    # get conditional, or mask
    mask =  X < 5
    cmask = X >= 10


    # get value according to mask
    xmask = tf.boolean_mask( X, mask)

    # change the value according to mask

    input_tensor = X
    to_be_changed_tensor = tf.ones_like( X ) * 10
    changeed_tensor = tf.where( mask, to_be_changed_tensor, input_tensor)

    conditional_indices = tf.where( mask )

    indices = conditional_indices[:,0]
    update_values = tf.ones(shape=tf.shape(X)[1], dtype=tf.int32) * 0
    update_array = tf.tile( update_values, [tf.shape(conditional_indices)[0]])
    update_array = tf.reshape( update_array, shape = [tf.shape( conditional_indices)[0],-1])
    print( indices, update_values, update_array)
    change_values = tf.scatter_update(X, indices, update_array )

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run( init )

        print('input value\n', sess.run(X))

        print('assign_mask\n', sess.run(change_values))





if __name__ == "__main__":
    # test_assign_from_where_conditoins()
    # test_change_value_spare_tensor()
    test_boolean_mask_scatter_update()
