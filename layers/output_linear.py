import tensorflow as tf
import tensorflow.contrib as tc
from layers.attention import additive_attention
from layers.params import Params

regularizer = tf.contrib.layers.l2_regularizer(scale = 3e-7)

def linear(input_size, g, m, position):
    with tf.variable_scope("output_layer"):
        w_p = tf.get_variable(
            "wp"+position, shape=(1, input_size),
            initializer=tf.contrib.layers.xavier_initializer(),
            trainable=True
        )
        p = tf.nn.softmax(tf.reduce_sum(tf.concat([g, m], -1) * w_p, -1), -1)
        return p


def answer_pointer(h_p, p_length, u_q, q_length, hidden_size, params, batch_size):
    with tf.variable_scope("answer_pointer"):

        v_q = tf.get_variable("v_q", shape=[hidden_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        v_q = tf.expand_dims(v_q, 0)
        v_q = tf.tile(v_q, [tf.shape(u_q)[1], 1])
        inputs = [u_q, v_q]
        scores = additive_attention(inputs, params[1], q_length, scope="question_pooling")
        init_state = tf.reduce_sum(tf.expand_dims(scores, -1)*u_q, 1)
        inputs = [h_p, init_state]
        p1_logits = additive_attention(inputs, params[0], p_length)
        cell = tc.rnn.GRUCell(hidden_size*2)
        scores = tf.reduce_sum(tf.expand_dims(p1_logits, -1)*h_p, 1)
        _, state = cell(scores, init_state)
        inputs = [h_p, state]
        p2_logits = additive_attention(inputs, params[0], p_length)
        return p1_logits, p2_logits


def layer_normalization(x, num_units=None, epsilon=1e-9, scope=None, reuse=None):
    if num_units is None:
        num_units = x.get_shape()[-1]
    with tf.variable_scope(scope, default_name="layer_norm", reuse=reuse):
        scale = tf.get_variable(
            "layer_norm_scale", shape=[num_units], dtype=tf.float32, initializer=tf.ones_initializer(), regularizer=regularizer)
        bias = tf.get_variable(
            "layer_norm_bias", shape=[num_units], dtype=tf.float32, initializer=tf.zeros_initializer(), regularizer=regularizer)
        mean = tf.reduce_mean(x, axis=-1, keep_dims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keep_dims=True)
        norm = (x - mean)*tf.rsqrt(variance + epsilon)
        return scale * norm + bias


# def attention_pooling(u, v_q, scope=None):
#     with tf.VariableScope(scope or "answer_pointer"):
#         shape_list = u.get_shape().as_list()
#         batch_size = shape_list[0]
#         m = shape_list[1]
#         hidden_size = shape_list[2]
#         w_u = tf.get_variable("w_u", shape=[hidden_size, hidden_size],
#                               initializer=tf.contrib.layers.xavier_initializer, trainable=True)
#         w_v = tf.get_variable("w_v", shape=[hidden_size, hidden_size],
#                               initializer=tf.contrib.layers.xavier_initializer, trainable=True)
#         v = tf.get_variable("v", shape=[hidden_size, 1],
#                             initializer=tf.contrib.layers.xavier_initializer, trainable=True)
#         q = tf.reshape(u, [-1, hidden_size])
#         s = tf.matmul(tf.nn.tanh(tf.matmul(q, w_u) + tf.matmul(v_q, w_v)), v)
#         s = tf.nn.softmax(tf.reshape(s, [-1, m]), -1)
#         return s
# def answer_pointer(u_q, h_p, scope=None):
#     with tf.VariableScope(scope or "answer_pointer"):
#         shape_list = u_q.get_shape().as_list()
#         batch_size = shape_list[0]
#         m = shape_list[1]
#         hidden_size = shape_list[2]
#         v_q = tf.get_variable("v_q", shape=[batch_size*m, hidden_size],
#                               initializer=tf.contrib.layers.xavier_initializer, trainable=True)
#         s = attention_pooling(u_q, v_q, scope="initial_hidden_state")
#         q = tf.reshape(u_q, [-1, hidden_size])
#         r_q = tf.reduce_sum(tf.reshape(tf.reshape(s, [-1, 1])*q, [-1, m, hidden_size]), 1)
#         a = attention_pooling(h_p, r_q, scope="start")
#         start = tf.argmax(a, 1)
#         ct = tf.reshape(a, [-1, 1])*tf.reshape(h_p, [-1, hidden_size])
#         ct = tf.reduce_sum(tf.reshape(ct, [batch_size, -1, hidden_size]), 1)
#
#         return r_q