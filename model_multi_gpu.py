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


class BiDAFModel_ngpus(object):
    """
    implement the Network structure described in https://arxiv.org/abs/1611.01603
    """
    def __init__(self, args, data):
        self.logger = logging.getLogger("BiDAF")
        self.hidden_size = args.hidden_size
        self.use_dropout = args.dropout_keep_prob < 1
        self.learning_rate = args.learning_rate
        self.ref_answers = []
        self.ref_contents = []

        self.max_p_length = args.max_p_len
        self.max_q_length = args.max_q_len
        self.max_a_length = args.max_a_len

        self.x, self.q, self.x_length, self.q_length, self.start, self.end, self.x_embed, self.q_embed, self.dropout_keep_prob = data

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
        self.contextual_embedding()
        self.attention_flow()
        self.modeling()
        self.output()
        self.compute_loss()

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
        # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def save(self, model_dir, model_prefix):
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info("Model save in {}, with prefix {}.".format(model_dir, model_prefix))

    def restore(self,model_dir, model_prefix):
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info("Model restore from {}, with prefix.".format(model_dir, model_prefix))

def training(args, dataset, vocab, result_dir=None, result_prefix=None):

    logger = logging.getLogger("BiDAF")
    def dev_content_answer(data_path):
        ref_contents = []
        ref_answers = []
        with open(data_path+".content", "r") as ref_content_files:
            for content in ref_content_files:
                ref_contents.append(content.strip().split())
        with open(data_path+".answer", "r") as ref_answer_files:
            for answer in ref_answer_files:
                ref_answers.append(''.join(answer.strip().split()))
        return ref_contents, ref_answers
    logger.info("Loading dev data...")
    ref_contents, ref_answers = dev_content_answer(args.dev_files)

    gpu_avaiables = [0, 1, 2]
    batch_size = args.batch_size
    with tf.device("/cpu:0"):
        tower_grads = []
        x = tf.placeholder(tf.int32, [None, None])
        q = tf.placeholder(tf.int32, [None, None])
        x_length = tf.placeholder(tf.int32, [None])
        q_length = tf.placeholder(tf.int32, [None])
        start = tf.placeholder(tf.int32, [None])
        end = tf.placeholder(tf.int32, [None])
        dropout_keep_prob = tf.placeholder(tf.float32)
        with tf.variable_scope('word_embedding'):
            word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(vocab.size(), vocab.embed_dim),
                initializer=tf.constant_initializer(vocab.embeddings),
                trainable=True
            )
            x_emb = tf.nn.embedding_lookup(word_embeddings, x)
            q_emb = tf.nn.embedding_lookup(word_embeddings, q)
        opt = tf.train.AdamOptimizer(args.learning_rate)
        sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        sess_config.gpu_options.allow_growth = True

        # sess = tf.Session(config=sess_config)
        with tf.Session(config=sess_config) as sess:
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(args.n_gpus):
                    with tf.device("/gpu:%d" % gpu_avaiables[i]):
                        with tf.name_scope("tower_%d" % gpu_avaiables[i]):
                            _x = x[i * batch_size:(i + 1) * batch_size]
                            _q = q[i * batch_size:(i + 1) * batch_size]
                            _x_length = x_length[i * batch_size:(i + 1) * batch_size]
                            _q_length = q_length[i * batch_size:(i + 1) * batch_size]

                            _start = start[i * batch_size:(i + 1) * batch_size]
                            _end = end[i * batch_size:(i + 1) * batch_size]
                            _x_emb = x_emb[i * batch_size:(i + 1) * batch_size]
                            _q_emb = q_emb[i * batch_size:(i + 1) * batch_size]
                            data = (_x, _q, _x_length, _q_length, _start, _end, _x_emb, _q_emb, dropout_keep_prob)
                            model = BiDAFModel_ngpus(args, data)
                            tf.get_variable_scope().reuse_variables()
                            model_loss = model.loss
                            p1, p2 = model.p1, model.p2
                            grads = opt.compute_gradients(model_loss)
                            tower_grads.append(grads)
            grads = average_gradients(tower_grads)
            train_op = opt.apply_gradients(grads)

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            pad_id = vocab.get_id(vocab.pad_token)
            max_rougeL = 0
            for epoch in range(1, args.epochs + 1):
                train_batches = dataset.get_batches('train', batch_size * args.n_gpus, pad_id, shuffle=True)
                total_num, total_loss = 0, 0
                log_every_n_batch, n_batch_loss = 50, 0

                for bitx, batch in enumerate(train_batches, 1):
                    feed_dict = {x: batch['content_ids'],
                                 q: batch['question_ids'],
                                 x_length: batch['content_length'],
                                 q_length: batch['question_length'],
                                 start: batch['start'],
                                 end: batch['end'],
                                 dropout_keep_prob: dropout_keep_prob}
                    _, loss = sess.run([train_op, model_loss], feed_dict)

                    total_loss += loss
                    total_num += 1
                    n_batch_loss += loss
                    if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                        logger.info('Average loss from batch {} to {} is {}'.format(
                            bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                        n_batch_loss = 0
                train_loss = 1.0 * total_loss / total_num
                logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

                logger.info('Evaluating the model for epoch {}'.format(epoch))

                eval_batches = dataset.get_batches('dev', batch_size * args.n_gpus, pad_id, shuffle=False)
                start_indices = []
                end_indices = []
                for bitx, batch in enumerate(eval_batches, 1):
                    feed_dict = {x: batch['content_ids'],
                                 q: batch['question_ids'],
                                 x_length: batch['content_length'],
                                 q_length: batch['question_length'],
                                 start: batch['start'],
                                 end: batch['end'],
                                 dropout_keep_prob: 1.0}
                    start_probs, end_probs, loss = sess.run([p1, p2, model_loss], feed_dict)
                    # print(len(start_probs))
                    start_probs = np.array(start_probs)
                    end_probs = np.array(end_probs)
                    total_loss += loss
                    total_num += 1
                    start_indices += np.argmax(start_probs, axis=1).tolist()
                    end_indices += np.argmax(end_probs, axis=1).tolist()


                rouge_eval = RougeL()
                bleu_eval = Bleu()
                pred_answers = []
                if result_prefix is not None and result_dir is not None:
                    with open('./data/' + '.answer', 'r') as ref_answer_files:
                        for answer in ref_answer_files:
                            ref_answers.append(''.join(answer.strip().split()))

                    with open('./data/' + 'test.content', 'r') as ref_content_files:
                        for content in ref_content_files:
                            ref_contents.append(content.strip().split())

                for i in range(len(start_indices)):
                    start_idx = start_indices[i]
                    end_idx = end_indices[i]
                    if end_idx < start_idx:
                        end_idx = start_idx + args.max_a_length
                    end_idx = np.minimum(end_idx, start_idx + args.max_a_length)
                    pred_answer = ''.join(ref_contents[i][start_idx:end_idx + 1])
                    if result_prefix is not None and result_dir is not None:
                        pred_answers.append(pred_answer)

                    rouge_eval.add_inst(pred_answer, ref_answers[i])
                    bleu_eval.add_inst(pred_answer, ref_answers[i])

                bleu_score = bleu_eval.get_score()
                rouge_score = rouge_eval.get_score()

                bleu_rouge = {'Bleu-4': bleu_score, 'Rouge-l': rouge_score}
                ave_loss = 1.0 * total_loss / total_num

                if result_prefix is not None and result_dir is not None:
                    logger.info('Test Bleu-4 :{}'.format(bleu_score))
                    logger.info('Test Rouge-l : {}'.format(rouge_score))
                    result_file = os.path.join(result_dir, result_prefix + '.txt')
                    with open(result_file, 'w') as fout:
                        fout.write('\n'.join(pred_answers))
                logger.info('Dev eval loss {}'.format(ave_loss))
                logger.info('Dev eval result: {}'.format(bleu_rouge))
                if math.isnan(bleu_rouge['Bleu-4']) or math.isnan(bleu_rouge['Rouge-l']):
                    logger.info("Dev eval is nan!")
                    continue
                if bleu_rouge['Rouge-l'] > max_rougeL:
                    saver.save(args.model_dir, "BIDAF")
                    max_rougeL = bleu_rouge['Rouge-l']



def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

