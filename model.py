import logging
import tensorflow as tf
from layers.rnn import rnn
from layers.attention import attention
from layers.output_linear import linear
import os
class BiDAFModel(object):
    """
    implement the Network structure described in https://arxiv.org/abs/1611.01603
    """
    def __init__(self, vocab, args):
        self.logger = logging.getLogger("BiDAF")
        self.vocab = vocab
        self.hidden_size = args.hidden_size
        self.batch_size = args.batch_size
        self.use_dropout = args.dropout_keep_prob < 1
        self.learning_rate = args.learning_rate
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.bulid_graph()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer)

    def bulid_graph(self):
        """
        build graph
        :return:
        """
        self.placeholders()
        self.word_embedding()
        self.contextual_embedding()
        self.attention_flow()
        self.modeling()
        self.output()
        self.compute_loss()

    def placeholders(self):
        """
        Placeholders
        :return:
        """
        self.x = tf.placeholder(tf.int32, shape=[None, None])
        self.q = tf.placeholder(tf.int32, shape=[None, None])
        self.x_length = tf.placeholder(tf.int32, shape=[None])
        self.q_length = tf.placeholder(tf.int32, shape=[None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.y1 = tf.placeholder(tf.int32, shape=[None])
        self.y2 = tf.placeholder(tf.int32, shape=[None])

    def word_embedding(self):
        """
        word embeddings
        :return:
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                "word_embeddings", shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True
            )
            self.x_embed = tf.nn.embedding_lookup(self.word_embeddings, self.x)
            self.q_embed = tf.nn.embedding_lookup(self.word_embeddings, self.q)

    def contextual_embedding(self):
        """
        contextual embedding
        :return:
        """
        with tf.variable_scope('paragraph_encoding'):
            self.h = rnn(self.x_embed, self.hidden_size, dropout_keep_prob=self.dropout_keep_prob)
        with tf.variable_scope('question_enconding'):
            self.u = rnn(self.q_embed, self.hidden_size, dropout_keep_prob=self.dropout_keep_prob)
        if self.use_dropout:
            self.h = tf.nn.dropout(self.h, self.dropout_keep_prob)
            self.u = tf.nn.dropout(self.u, self.dropout_keep_prob)

    def attention_flow(self):
        """
        Attention Flow Layer
        contains Context-to-query Attention and Query-to-context Attention
        :return:
        """
        self.g = attention(self.batch_size, self.hidden_size, self.h, self.u)
        if self.use_dropout:
            self.g = tf.nn.dropout(self.g, self.dropout_keep_prob)

    def modeling(self):
        self.m = rnn(self.g, self.hidden_size, self.x_length, layer_num=2, dropout_keep_prob=self.dropout_keep_prob)
        if self.use_dropout:
            self.m = tf.nn.dropout(self.m, self.dropout_keep_prob)

    def output(self):
        self.p1 = linear(self.batch_size, self.hidden_size, self.g, self.m, '1')
        m_ = rnn(self.m, self.hidden_size, self.x_length, dropout_keep_prob=self.dropout_keep_prob)
        self.p2 = linear(self.batch_size, self.hidden_size, self.g, m_, '2')

    def compute_loss(self):
        def log_loss(probs, y):
            with tf.name_scope("log_loss"):
                y = tf.one_hot(y, tf.shape(probs)[1], axis=1)
                loss = - tf.reduce_sum(probs * tf.log(probs), 1)
                return loss
        self.start_loss = log_loss(self.p1, self.y1)
        self.end_loss = log_loss(self.p2, self.y2)
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def save(self, model_dir, model_prefix):
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info("Model save in {}, with prefix {}.".format(model_dir, model_prefix))

    def restore(self,model_dir, model_prefix):
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info("Model restore from {}, with prefix.".format(model_dir, model_prefix))

