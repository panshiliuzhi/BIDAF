import tensorflow as tf


def linear(batch_size, hidden_size, g, m, position):
    with tf.variable_scope("output_layer"):
        w_p = tf.get_variable(
            "wp"+position, shape=(batch_size, hidden_size*10, 1),
            initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0),
            trainable=True
        )
        p = tf.nn.softmax(tf.squeeze(tf.matmul(tf.concat([g, m], -1), w_p),[-1]), -1)
        return p