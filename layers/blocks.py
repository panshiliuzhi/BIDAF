import tensorflow as tf
import tensorflow.contrib as tc
from layers.rnn import linear
from layers.output_linear import layer_normalization
initializer = lambda: tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
initializer_relu = lambda: tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                             mode='FAN_IN',
                                                             uniform=False,
                                                             dtype=tf.float32)
regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)


def residual_block(inputs, num_blocks=1, input_size=128, conv_nums=1, num_filters=128, head=8, projection=True, scope=None, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        if projection:
            batch_size = tf.shape(inputs)[0]
            inputs = tf.reshape(inputs, [-1, tf.shape(inputs)[-1]])
            inputs = linear(inputs, input_size, num_filters, scope="projection")
            inputs = tf.reshape(inputs, [batch_size, -1, num_filters])
        inputs += position_encoding(inputs, num_filters)
        outputs = inputs
        for j in range(num_blocks):
            outputs = tf.expand_dims(outputs, 2)
            for i in range(conv_nums):
                outputs = layer_normalization(outputs, num_filters, scope="conv_layer_norm%d_%d"%(j, i), reuse=reuse)
                outputs = depthwise_separable_conv(outputs, scope="depthwise_separable_conv%d_%d"%(j, i), reuse=reuse) + outputs
            outputs = tf.squeeze(outputs, 2)
            outputs = layer_normalization(outputs, num_filters, scope="conv_layer_norm%d" %j, reuse=reuse)
            outputs = multi_head_attention(outputs, outputs, outputs, num_filters, head, scope="multi_head_attention%d"%j, reuse=reuse) + outputs
        return outputs


def depthwise_separable_conv(inputs, kernel_size=7, num_filters=128, bias=True, reuse=None, scope="depthwise_separable_conv"):
    with tf.variable_scope(scope, reuse=reuse):
        shapes = inputs.shape.as_list()

        depthwise_filter = tf.get_variable("depthwise_filter",
                                           shape=[kernel_size, 1, shapes[-1], 1], dtype=tf.float32, initializer=initializer_relu()
                                           ,trainable=True, regularizer=regularizer)
        pointwise_filter = tf.get_variable("pointwise_filter",
                                           shape=[1, 1, shapes[-1], num_filters], dtype=tf.float32, initializer=initializer_relu(),
                                           trainable=True, regularizer=regularizer)
        outputs = tf.nn.separable_conv2d(inputs, depthwise_filter, pointwise_filter, strides=[1, 1, 1, 1], padding="SAME")
        if bias:
            b = tf.get_variable("bias",
                                outputs.shape[-1], regularizer=regularizer, initializer=tf.zeros_initializer())
            outputs += b
        outputs = tf.nn.relu(outputs)
        return outputs


def position_encoding(inputs, position_size):
    batch_size, length = tf.shape(inputs)[0], tf.shape(inputs)[1]
    position_i = 1./tf.pow(1000., 2* tf.range(position_size/2, dtype=tf.float32)/position_size)
    position_i = tf.expand_dims(position_i, 0)
    position_j = tf.range(tf.cast(length, tf.float32), dtype=tf.float32)
    position_j = tf.expand_dims(position_j, 1)
    position = tf.matmul(position_j, position_i)
    position = tf.concat([tf.cos(position), tf.sin(position)], -1)
    position = tf.expand_dims(position, 0) + tf.zeros((batch_size, length, position_size), dtype=tf.float32)
    return position


def multi_head_attention(Q, K, V, d, head, scope="Multi_Head_Attention", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        batch_size, length = tf.shape(Q)[0], tf.shape(Q)[1]
        Q = linear(Q, d, d*head, scope="Q")
        Q = tf.reshape(Q, [-1, length, head, d])
        Q = tf.transpose(Q, [0, 2, 1, 3])
        #batch_size, length = tf.shape(K)[0], tf.shape(K)[1]
        K = linear(K, d, d*head, scope="K")
        K = tf.reshape(K, [-1, length, head, d])
        K = tf.transpose(K, [0, 2, 1, 3])
        S = tf.matmul(Q, K, transpose_b=True)/ tf.sqrt(float(d))
        S = tf.nn.softmax(S)
        #batch_size, length = tf.shape(V)[0], tf.shape(V)[1]
        V = linear(V, d, d*head, scope="V")
        V = tf.reshape(V, [-1, length, head, d])
        V = tf.transpose(V, [0, 2, 1, 3])
        output = tf.matmul(S, V)
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, [batch_size*length, -1])
        output = linear(output, d*head, d, scope="linear_output")
        output = tf.reshape(output, [batch_size, -1, tf.shape(output)[-1]])
        return output