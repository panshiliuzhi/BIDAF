import tensorflow as tf
import tensorflow.contrib as tc
from layers.attention import additive_attention
from tensorflow.contrib.rnn import RNNCell

from tensorflow.python.util import nest

def rnn(rnn_type, inputs, hidden_size, length, layer_num=1, dropout_keep_prob=None):
    cell_fw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
    cell_bw = get_cell(rnn_type, hidden_size, layer_num, dropout_keep_prob)
    # outputs 包含(output_fw, output_bw)，其中每一个都是[batch_size, max_time, cell_fw.output_size]
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, inputs, sequence_length=length, dtype=tf.float32)
    return tf.concat(outputs, 2)


def get_cell(rnn_type, hidden_size, layer_num=1, dropout_keep_prob=None):
    cells = []
    for i in range(layer_num):
        if rnn_type == "lstm" :
            cell = tc.rnn.LSTMCell(hidden_size, state_is_tuple=True)
        elif rnn_type == "gru":
            cell = tc.rnn.GRUCell(hidden_size)
        elif rnn_type == "rnn":
            cell = tc.rnn.BasicRNNCell(hidden_size)
        else:
            raise NotImplementedError('Unsuported rnn type: {}'.format(rnn_type))
        if dropout_keep_prob is not None:
            cell = tc.rnn.DropoutWrapper(cell, input_keep_prob=dropout_keep_prob, output_keep_prob=dropout_keep_prob)
        cells.append(cell)
    cells = tc.rnn.MultiRNNCell(cells, state_is_tuple=True)
    return cells


def gatedAttentionGRU(memory, memory_len, inputs, length, hidden_size, params, use_state=False, dropout_keep_prob=None):
    cell = custom_GRU_cell(hidden_size, memory, memory_len, params, use_state)
    if dropout_keep_prob is not None:
        cell = tc.rnn.DropoutWrapper(cell, input_keep_prob=dropout_keep_prob, output_keep_prob=dropout_keep_prob)
    cell_fw , cell_bw= cell, cell
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, length, dtype=tf.float32)
    return tf.concat(outputs, 2)


class custom_GRU_cell(RNNCell):

    def __init__(self, num_units, memory, memory_len, params, use_state):
        super(custom_GRU_cell, self).__init__(_reuse=None)
        self._num_units = num_units
        self.memory = memory
        self.use_state = use_state
        self.memory_len = memory_len
        self.params = params
    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        weights, w_g = self.params
        inputs_ = [self.memory, inputs]
        if self.use_state:
            inputs_.append(state)
        scores = additive_attention(inputs_, weights, self.memory_len)
        scores = tf.expand_dims(scores, -1)
        ct = tf.reduce_sum(scores*self.memory, 1)
        # input = tf.cast(input, tf.float32)
        # ct = tf.cast(ct, tf.float32)
        inputs = tf.concat([inputs, ct], -1)
        gt = tf.sigmoid(tf.matmul(inputs, w_g))
        inputs = gt*inputs
        # if inputs.get_shape().as_list()[-1] != self._num_units:
        #     with tf.variable_scope("projection"):
        #         res = _linear(inputs, self._num_units, False)
        # else:
        #     res = inputs
        with tf.variable_scope("Gates"):
            r, u = tf.split(_linear([inputs, state], 2*self._num_units, True, 1.0), num_or_size_splits=2, axis=1)
            r, u = tf.nn.sigmoid(r), tf.nn.sigmoid(u)
        with tf.variable_scope("Candidate"):
            c = tf.nn.tanh(_linear([inputs, r*state], self._num_units, True))
        new_h = u * state + (1 - u)*c
        return new_h, new_h

def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        ars = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
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


def linear(inputs, input_size, output_size, bias=None, bias_start=0.0, scope=None):
    shapes = inputs.get_shape().as_list()
    if len(shapes) > 2:
        inputs = tf.reshape(inputs, [-1, tf.shape(inputs)[-1]])
    with tf.variable_scope(scope or "linear"):
        matrix = tf.get_variable(scope+"Matrix", [input_size, output_size], dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    res = tf.matmul(inputs, matrix)
    if not bias :
        return res
    bias_term = tf.get_variable(
        scope+"Bias", [output_size], dtype=tf.float32, initializer=tf.constant_initializer(bias_start, dtype=tf.float32)
    )
    return res + bias_term

def get_attn_params(hidden_size, initializer=tf.truncated_normal_initializer):
    with tf.variable_scope("attention_weights"):
        params = {
            "w_u_q":tf.get_variable("w_u_q", shape=[hidden_size*2, hidden_size], dtype=tf.float32, initializer=initializer(), trainable=True),
            "w_u_p":tf.get_variable("w_u_p", shape=[hidden_size*2, hidden_size], dtype=tf.float32, initializer=initializer(), trainable=True),
            "w_v_p":tf.get_variable("w_v_p", shape=[hidden_size, hidden_size], dtype=tf.float32, initializer=initializer(), trainable=True),
            "w_g":tf.get_variable("w_g", shape=[hidden_size*4, hidden_size*4], dtype=tf.float32, initializer=initializer(), trainable=True),
            "w_v_p_":tf.get_variable("w_v_p_", shape=[hidden_size*2, hidden_size], dtype=tf.float32, initializer=initializer(), trainable=True),
            "w_v_phat":tf.get_variable("w_v_phat",  shape=[hidden_size*2, hidden_size], dtype=tf.float32, initializer=initializer(), trainable=True),
            "w_g_":tf.get_variable("w_g_", shape=[hidden_size*4, hidden_size*4], dtype=tf.float32, initializer=initializer(), trainable=True),
            "w_h_p":tf.get_variable("w_h_p", shape=[hidden_size*2, hidden_size], dtype=tf.float32, initializer=initializer(), trainable=True),
            "w_h_a":tf.get_variable("w_h_a", shape=[hidden_size*2, hidden_size], dtype=tf.float32, initializer=initializer(), trainable=True),
            "w_v_q":tf.get_variable("w_v_q", shape=[hidden_size, hidden_size], dtype=tf.float32, initializer=initializer(), trainable=True),
            "w_u_q_":tf.get_variable("w_u_q_", shape=[hidden_size*2, hidden_size], dtype=tf.float32, initializer=initializer(), trainable=True),
            "v":tf.get_variable("v", shape=[hidden_size], dtype=tf.float32, initializer=initializer(), trainable=True)
            # "v2": tf.get_variable("v2", shape=[hidden_size], dtype=tf.float32, initializer=initializer(),
            #                        trainable=True),
            # "v3": tf.get_variable("v3", shape=[hidden_size], dtype=tf.float32, initializer=initializer(),
            #                        trainable=True),
            # "v4": tf.get_variable("v4", shape=[hidden_size], dtype=tf.float32, initializer=initializer(),
            #                       trainable=True)
        }
        return params

