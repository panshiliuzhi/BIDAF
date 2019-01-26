import tensorflow as tf
import tensorflow.contrib as tc


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
