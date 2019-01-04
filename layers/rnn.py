import tensorflow as tf
import tensorflow.contrib as tc


def rnn(inputs, hidden_size, length, layer_num=1, dropout_keep_prob=None):
    cell_fw = get_cell(hidden_size, layer_num, dropout_keep_prob)
    cell_bw = get_cell(hidden_size, layer_num, dropout_keep_prob)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, inputs, sequence_length=length, dtype=tf.float32)
    return tf.concat(outputs, 2)


def get_cell(hidden_size, layer_num=1, dropout_keep_prob=None):
    cells = []
    for i in range(layer_num):
        cell = tc.rnn.LSTMCell(hidden_size, state_is_tuple=True)
        if dropout_keep_prob is not None:
            cell = tc.rnn.DropoutWrapper(cell, input_keep_prob=dropout_keep_prob, output_keep_prob=dropout_keep_prob)
        cells.append(cell)
    cells = tc.rnn.MultiRNNCell(cells, state_is_tuple=True)
    return cells
