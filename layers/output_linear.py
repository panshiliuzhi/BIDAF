import tensorflow as tf
import tensorflow.contrib as tc


def linear(hidden_size, g, m, position):
    with tf.variable_scope("output_layer"):
        w_p = tf.get_variable(
            "wp"+position, shape=(1, hidden_size*10),
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True
        )
        p = tf.nn.softmax(tf.reduce_sum(tf.concat([g, m], -1) * w_p, -1), -1)
        return p