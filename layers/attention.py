import tensorflow as tf

from tensorflow.python.util import nest
def content_attention(input_size, h, u):
    """

    :param h: passae_encodes
    :param u: question_encodes
    :return:
    """
    with tf.variable_scope('attention_flow'):
        # method one
        # h_aug = tf.tile(tf.expand_dims(h, 2), [1, 1, J, 1])
        # u_aug = tf.tile(tf.expand_dims(u, 1), [1, T, 1, 1])
        # s = trilinear([h_aug, u_aug, h_aug*u_aug])
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
            name="w_s", shape=(1, input_size),
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


def mask_attenion_score(scores, memory_len, mask_value=-1e8):
    score_mask = tf.sequence_mask(memory_len, maxlen=scores.shape.as_list()[1])
    score_mask_value = tf.ones_like(scores)*mask_value
    #print(scores.shape.as_list(), score_mask_value.shape.as_list())
    return tf.where(score_mask, scores, score_mask_value)


def additive_attention(inputs, params, memory_len=None, scope="attention"):

    with tf.variable_scope(scope):
        weights, v = params
        outputs = []
        for i, (up, w) in enumerate(zip(inputs, weights)):
            axis_num = len(up.shape.as_list())
            shapes = tf.shape(up) #
            up = tf.reshape(up, (-1, shapes[-1]))
            output = tf.matmul(up, w)
            if axis_num > 2:
                output = tf.reshape(output, (shapes[0], shapes[1], -1))
            elif axis_num == 2 and scope == "attention" :
                output = tf.reshape(output, (shapes[0], 1, -1))
            elif scope == "question_pooling":
                output = tf.reshape(output, (1, shapes[0], -1))
            outputs.append(output)
        outputs = sum(outputs)
        scores = tf.reduce_sum(tf.nn.tanh(outputs)*v, -1)
    # if memory_len is not None:
    #     scores = mask_attenion_score(scores, memory_len)
    return tf.nn.softmax(scores)


def trilinear(args, output_size=1, bias=True, scope="trilinear"):
    with tf.variable_scope(scope):
        num_units = tf.shape(args)[-1]
        flatten_args = [tf.reshape(arg, [-1, num_units]) for arg in args]
        out = _linear(flatten_args, output_size, bias=bias, scope=scope)
        return tf.squeeze(out, -1)

def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        ars = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        # if len(shape) != 2:
        #     raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        # if not shape[1]:
        #     raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        # else:
        total_arg_size += shape[1]
    dtype = [a.dtype for a in args][0]
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable(
            "Matrix", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(args, 1), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size], dtype=dtype, initializer=tf.constant_initializer(bias_start, dtype=dtype)
        )
        return res + bias_term






