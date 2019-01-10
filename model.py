import logging
import tensorflow as tf
from layers.rnn import rnn
from layers.attention import attention
from layers.output_linear import linear
import os
import time
import numpy as np
import math
from evaluate.bleu import Bleu
from evaluate.rouge import RougeL


class BiDAFModel(object):
    """
    implement the Network structure described in https://arxiv.org/abs/1611.01603
    """
    def __init__(self, vocab, args):
        self.logger = logging.getLogger("BiDAF")
        self.vocab = vocab
        self.hidden_size = args.hidden_size
        self.use_dropout = args.dropout_keep_prob < 1
        self.learning_rate = args.learning_rate
        self.ref_answers = []
        self.ref_contents = []

        self.max_p_length = args.max_p_len
        self.max_q_length = args.max_q_len
        self.max_a_length = args.max_a_len

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.bulid_graph()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def bulid_graph(self):
        """
        build graph
        :return:
        """
        start_time = time.time()
        self.placeholders()
        self.word_embedding()
        self.contextual_embedding()
        self.attention_flow()
        self.modeling()
        self.output()
        self.compute_loss()
        self.logger.info("Time to build graph: {} s".format(time.time() - start_time))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info("There are {} parameters in the model".format(param_num))

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
        self.start = tf.placeholder(tf.int32, shape=[None])
        self.end = tf.placeholder(tf.int32, shape=[None])

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
            self.h = rnn(self.x_embed, self.hidden_size, self.x_length)
        with tf.variable_scope('question_enconding'):
            self.u = rnn(self.q_embed, self.hidden_size, self.q_length)
        if self.use_dropout:
            self.h = tf.nn.dropout(self.h, self.dropout_keep_prob)
            self.u = tf.nn.dropout(self.u, self.dropout_keep_prob)

    def attention_flow(self):
        """
        Attention Flow Layer
        contains Context-to-query Attention and Query-to-context Attention
        :return:
        """
        self.g = attention(self.hidden_size, self.h, self.u)
        if self.use_dropout:
            self.g = tf.nn.dropout(self.g, self.dropout_keep_prob)

    def modeling(self):
        with tf.variable_scope("modeling"):
            self.m = rnn(self.g, self.hidden_size, self.x_length, layer_num=1)
        if self.use_dropout:
            self.m = tf.nn.dropout(self.m, self.dropout_keep_prob)

    def output(self):
        self.p1 = linear(self.hidden_size, self.g, self.m, '1')
        with tf.variable_scope("output_rnn"):
            m_ = rnn(self.m, self.hidden_size, self.x_length, layer_num=1)
        self.p2 = linear(self.hidden_size, self.g, m_, '2')

    def compute_loss(self):
        def log_loss(probs, y, epsion = 1e-9):
            with tf.name_scope("log_loss"):
                y = tf.one_hot(y, tf.shape(probs)[1], axis=1)
                loss = - tf.reduce_sum(y * tf.log(probs + epsion), 1)
                return loss
        self.start_loss = log_loss(self.p1, self.start)
        self.end_loss = log_loss(self.p2, self.end)
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train_one_epoch(self, batch_datas, dropout_keep_prob):
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        for bitx, batch in enumerate(batch_datas, 1):
            feed_dict = {
                self.x: batch['content_ids'],
                self.q: batch['question_ids'],
                self.x_length: batch['content_length'],
                self.q_length: batch['question_length'],
                self.start: batch['start'],
                self.end: batch['end'],
                self.dropout_keep_prob: dropout_keep_prob
            }
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            # print("start: ", start)
            # print("end: ", end)
            # print("loss:", loss)
            total_loss += loss
            total_num += 1
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss/log_every_n_batch
                ))
                n_batch_loss = 0
        return 1.0*total_loss/total_num

    def train(self, data, epochs, batch_size, save_dir=None, save_prefix=None, dropout_keep_prob=1.0, evaluate=True):
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        max_rougeL = 0
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            batch_datas = data.get_batches('train', batch_size, pad_id, shuffle=True)
            train_loss = self.train_one_epoch(batch_datas, dropout_keep_prob)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.get_batches('dev', batch_size, pad_id, shuffle=False)
                    eval_loss, bleu_rouge = self.evaluate(eval_batches)
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))
                    if math.isnan(bleu_rouge['Bleu-4']) or math.isnan(bleu_rouge['Rouge-l']):
                        self.logger.info("Dev eval is nan!")
                        continue
                    if bleu_rouge['Rouge-l'] > max_rougeL:
                        self.save(save_dir, save_prefix)
                        max_rougeL = bleu_rouge['Rouge-l']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """


        total_loss, total_num = 0, 0
        start_indices = []
        end_indices = []
        for b_itx, batch in enumerate(eval_batches):
            feed_dict = {self.x: batch['content_ids'],
                         self.q: batch['question_ids'],
                         self.x_length: batch['content_length'],
                         self.q_length: batch['question_length'],
                         self.start: batch['start'],
                         self.end: batch['end'],
                         self.dropout_keep_prob: 1.0}
            # print(feed_dict)
            start_probs, end_probs, loss = self.sess.run([self.p1,
                                                          self.p2, self.loss], feed_dict)
            #print(len(start_probs))
            start_probs = np.array(start_probs)
            end_probs = np.array(end_probs)
            total_loss += loss
            total_num += 1
            start_indices += np.argmax(start_probs,axis=1).tolist()
            end_indices += np.argmax(end_probs,axis=1).tolist()
            # print(len(start_indices))



        rouge_eval = RougeL()
        bleu_eval = Bleu()
        pred_answers = []
        if result_prefix is not None and result_dir is not None:
            with open('./data/'+'.answer', 'r') as ref_answer_files:
                for answer in ref_answer_files:
                    self.ref_answers.append(''.join(answer.strip().split()))

            with open('./data/'+'test.content', 'r') as ref_content_files:
                for content in ref_content_files:
                    self.ref_contents.append(content.strip().split())

        for i in range(len(start_indices)):
            start_idx = start_indices[i]
            end_idx = end_indices[i]
            if end_idx < start_idx:
                end_idx = start_idx + self.max_a_length
            end_idx = np.minimum(end_idx, start_idx+self.max_a_length)
            pred_answer = ''.join(self.ref_contents[i][start_idx:end_idx+1])
            if result_prefix is not None and result_dir is not None:
                pred_answers.append(pred_answer)

            rouge_eval.add_inst(pred_answer, self.ref_answers[i])
            bleu_eval.add_inst(pred_answer, self.ref_answers[i])

        bleu_score = bleu_eval.get_score()
        rouge_score = rouge_eval.get_score()

        bleu_rouge = {'Bleu-4':bleu_score,'Rouge-l':rouge_score}
        ave_loss = 1.0 * total_loss / total_num

        if result_prefix is  not None and  result_dir is not None:
            self.logger.info('Test Bleu-4 :{}'.format(bleu_score))
            self.logger.info('Test Rouge-l : {}'.format(rouge_score))
            result_file = os.path.join(result_dir, result_prefix + '.txt')
            with open(result_file, 'w') as fout:
                    fout.write('\n'.join(pred_answers))

        return ave_loss, bleu_rouge

    def save(self, model_dir, model_prefix):
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info("Model save in {}, with prefix {}.".format(model_dir, model_prefix))

    def restore(self,model_dir, model_prefix):
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info("Model restore from {}, with prefix.".format(model_dir, model_prefix))

    def dev_content_answer(self, data_path):
        with open(data_path+".content", "r") as ref_content_files:
            for content in ref_content_files:
                self.ref_contents.append(content.strip().split())
        with open(data_path+".answer", "r") as ref_answer_files:
            for answer in ref_answer_files:
                self.ref_answers.append(''.join(answer.strip().split()))






