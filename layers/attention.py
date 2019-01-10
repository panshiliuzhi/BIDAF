import tensorflow as tf


def attention(hidden_size, h, u):
    """

    :param h: passae_encodes
    :param u: question_encodes
    :return:
    """
    with tf.variable_scope('attention_flow'):
        T = tf.shape(h)[1]
        J = tf.shape(u)[1]

        # method one
        # h_aug = tf.tile(tf.expand_dims(h, 2), [1, 1, J, 1])
        # u_aug = tf.tile(tf.expand_dims(u, 1), [1, T, 1, 1])
        # s = tf.concat([h_aug, u_aug, h_aug*u_aug], -1)
        # s = tf.reshape(s, [batch_size, -1, tf.shape(s)[-1]])
        # w_s = tf.get_variable(
        #     "w_s", shape=(batch_size, hidden_size*6, 1),
        #     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0),
        #     trainable=True
        # )
        # s = tf.reshape(tf.squeeze(tf.matmul(s, w_s), [-1]), [batch_size, T, J])

        # method two
        w = tf.get_variable(
            name="w_s", shape=(1, hidden_size * 2),
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True
        )
        h = h * w
        s = tf.matmul(h, u, transpose_b=True)
        u_a = tf.matmul(tf.nn.softmax(s, -1), u)
        b = tf.expand_dims(tf.nn.softmax(tf.reduce_max(s, 2), 1), 1)
        h_b = tf.tile(tf.matmul(b, h), [1, tf.shape(h)[1], 1])
        G = tf.concat([h, u_a, h * u_a, h * h_b], -1)
    return G






