import logging
import tensorflow as tf
import os
import time
import numpy as np
import math
from evaluate.bleu import Bleu
from evaluate.rouge import RougeL
from layers.rnn import rnn
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings



class RNETModel(object):
    def __init__(self, args):

        self.hidden_size = args.hidden_size
        self.use_dropout = args.dropout_keep_prob < 1
        self.logger = logging.getLogger("BiDAF")
        data_dir = '/home/home1/dmyan/codes/bilm-tf/bilm/data/models/'
        self.options_file = data_dir + 'options.json'
        self.weight_file = data_dir + 'weights.hdf5'
        self.token_embedding_file = data_dir + 'vocab_embedding.hdf5'

    def placeholders(self):
        self.W_Q = tf.placeholder(tf.int32, shape=[None, None])
        self.W_P = tf.placeholder(tf.int32, shape=[None, None])
        self.q_length = tf.placeholder(tf.int32, shape=[None])
        self.p_length = tf.placeholder(tf.int32, shape=[None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.start = tf.placeholder(tf.int32, shape=[None])
        self.end = tf.placeholder(tf.int32, shape=[None])

    def word_embedding(self):
        bilm = BidirectionalLanguageModel(
            self.options_file,
            self.weight_file,
            use_character_inputs=False,
            embedding_weight_file=self.token_embedding_file
        )
        context_embeddings_op = bilm(self.W_P)
        question_embeddings_op = bilm(self.W_Q)

        elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
        with tf.variable_scope('', reuse=True):
            # the reuse=True scope reuses weights from the context for the question
            elmo_question_input = weight_layers(
                'input', question_embeddings_op, l2_coef=0.0
            )
        self.p_embed, self.q_embed= elmo_context_input['weighted_op'], elmo_question_input['weighted_op']

    def encode(self):
        with tf.variable_scope("passage_encoding"):
            self.u_p = rnn('gru', self.p_embed, self.hidden_size, self.p_length, layer_num=3)
        with tf.variable_scope("question_encoding"):
            self.u_q = rnn('gru', self.q_embed, self.hidden_size, self.q_length, layer_num=3)
        if self.use_dropout:
            self.u_p = tf.nn.dropout(self.u_p, self.dropout_keep_prob)
            self.u_q = tf.nn.dropout(self.u_q, self.dropout_keep_prob)
    def gated_attention(self):
        tf.nn.dynamic_rnn


