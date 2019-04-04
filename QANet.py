import tensorflow as tf
import logging
from layers.blocks import residual_block
from layers.attention import content_attention
import os
import time
import numpy as np
import math
from evaluate.bleu import Bleu
from evaluate.rouge import RougeL
from layers.blocks import multi_head_attention
from layers.output_linear import linear
from bilm import  BidirectionalLanguageModel, weight_layers

class QANetModel(object):
    def __init__(self, args):
        self.logger = logging.getLogger(args.algo)
        self.hidden_size = args.hidden_size
        self.use_dropout = args.dropout_keep_prob < 1
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.head_count = args.head_count
        data_dir = '/home/home1/dmyan/codes/bilm-tf/bilm/data/models/'
        self.options_file = data_dir + 'options.json'
        self.weight_file = data_dir + 'weights.hdf5'
        self.token_embedding_file = data_dir + 'vocab_embedding.hdf5'
        self.ref_answers = []
        self.ref_contents = []
        self.max_a_length = args.max_a_len
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.build_graph()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
    def build_graph(self):
        start_time = time.time()
        self.placeholders()
        self.word_embedding()
        self.encoder()
        self.attention_flow()
        self.model_encoder()
        self.output()
        self.compute_loss()
        self.logger.info("Time to build graph: {} s".format(time.time() - start_time))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info("There are {} parameters in the model".format(param_num))

    def placeholders(self):
        self.W_Q = tf.placeholder(tf.int32, shape=[None, None])
        self.W_P = tf.placeholder(tf.int32, shape=[None, None])
        self.q_length = tf.placeholder(tf.int32, shape=[None])
        self.p_length = tf.placeholder(tf.int32, shape=[None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.start = tf.placeholder(tf.int32, shape=[None])
        self.end = tf.placeholder(tf.int32, shape=[None])

    def word_embedding(self):
        #with tf.variable_scope("word_embedding_layer"):
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

    def encoder(self):
        with tf.variable_scope("embedding_encoder_layer"):
            self.p_encode = residual_block(self.p_embed, 1, 300, conv_nums=4, head=4, scope="Encoder_Block")
            self.q_encode = residual_block(self.q_embed, 1, 300, conv_nums=4, head=4, scope="Encoder_Block", reuse=True)
            if self.use_dropout:
                self.p_encode = tf.nn.dropout(self.p_encode, self.dropout_keep_prob)
                self.q_encode = tf.nn.dropout(self.q_encode, self.dropout_keep_prob)
    def attention_flow(self):
        with tf.variable_scope("attention_flow_layer"):
            #self.g = multi_head_attention(self.q_encode, self.p_encode, self.p_encode, self.hidden_size, 8, scope="question_content_attention")
            self.g = content_attention(self.hidden_size, self.p_encode, self.q_encode)
            if self.use_dropout:
                self.g = tf.nn.dropout(self.g, self.dropout_keep_prob)
    def model_encoder(self):
        with tf.variable_scope("model_encoder_layer"):
            self.enc = [self.g]
            for i in range(3):
                self.enc.append(
                    residual_block(
                        self.enc[i],
                        num_blocks=4,
                        input_size=self.hidden_size*4,
                        conv_nums=2,
                        head=4,
                        projection= True if i == 0 else False
                        ,scope="model_encoder",
                        reuse=True if i > 0 else None
                    )
                )
    def output(self):
        with tf.variable_scope("output_layer"):
            self.p1 = linear(self.hidden_size*2, self.enc[1], self.enc[2], position="1")
            self.p2 = linear(self.hidden_size*2, self.enc[1], self.enc[3], position="2")


    def compute_loss(self):
        def log_loss(probs, y, epsion = 1e-9):
            with tf.name_scope("log_loss"):
                y = tf.one_hot(y, tf.shape(probs)[1], axis=1)
                loss = - tf.reduce_sum(y * tf.log(probs + epsion), 1)
                return loss
        start_loss = log_loss(self.p1, self.start)
        end_loss = log_loss(self.p2, self.end)
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(start_loss, end_loss))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss, self.all_params)
        for i, (g, v) in enumerate(grads_and_vars):
            if g is not None:
                grads_and_vars[i] = (tf.clip_by_norm(g, 10), v)
        self.train_op = optimizer.apply_gradients(grads_and_vars)#tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def train_one_epoch(self, batch_datas, dropout_keep_prob):
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        for bitx, batch in enumerate(batch_datas, 1):
            feed_dict = {
                self.W_P: batch['content_ids'],
                self.W_Q: batch['question_ids'],
                self.p_length: batch['content_length'],
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
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch
                ))
                n_batch_loss = 0
        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir=None, save_prefix=None, dropout_keep_prob=1.0, evaluate=True):
        pad_id = 0
        max_rougeL = 0.87
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
        evaluate after one epoch
        :param eval_batches:
        :param result_dir:
        :param result_prefix:
        :return:
        """

        total_loss, total_num = 0, 0
        start_indices = []
        end_indices = []
        for b_itx, batch in enumerate(eval_batches):
            feed_dict = {
                self.W_P: batch['content_ids'],
                self.W_Q: batch['question_ids'],
                self.p_length: batch['content_length'],
                self.q_length: batch['question_length'],
                self.start: batch['start'],
                self.end: batch['end'],
                self.dropout_keep_prob: 1.0}
            # print(feed_dict)
            start_probs, end_probs, loss = self.sess.run([self.p1,
                                                          self.p2, self.loss], feed_dict)
            # print(len(start_probs))
            start_probs = np.array(start_probs)
            end_probs = np.array(end_probs)
            total_loss += loss
            total_num += 1
            start_indices += np.argmax(start_probs, axis=1).tolist()
            end_indices += np.argmax(end_probs, axis=1).tolist()
            # print(len(start_indices))

        rouge_eval = RougeL()
        bleu_eval = Bleu()
        pred_answers = []
        if result_prefix is not None and result_dir is not None:
            with open('./data/' + 'test.answer', 'r') as ref_answer_files:
                for answer in ref_answer_files:
                    self.ref_answers.append(''.join(answer.strip().split()))

            with open('./data/' + 'test.content', 'r') as ref_content_files:
                for content in ref_content_files:
                    self.ref_contents.append(content.strip().split())

        for i in range(len(start_indices)):
            start_idx = start_indices[i]
            end_idx = end_indices[i]
            if end_idx < start_idx:
                end_idx = start_idx + self.max_a_length
            end_idx = np.minimum(end_idx, start_idx + self.max_a_length)
            pred_answer = ''.join(self.ref_contents[i][start_idx:end_idx + 1])
            if result_prefix is not None and result_dir is not None:
                pred_answers.append(pred_answer)

            rouge_eval.add_inst(pred_answer, self.ref_answers[i])
            bleu_eval.add_inst(pred_answer, self.ref_answers[i])

        bleu_score = bleu_eval.get_score()
        prec, rec, rouge_score = rouge_eval.get_score()
        self.logger.info("Evaluate precision, recall is : {}, {}".format(prec, rec))
        bleu_rouge = {'Bleu-4': bleu_score, 'Rouge-l': rouge_score}
        ave_loss = 1.0 * total_loss / total_num

        if result_prefix is not None and result_dir is not None:
            self.logger.info('Test Bleu-4 :{}'.format(bleu_score))
            self.logger.info('Test Rouge-l : {}'.format(rouge_score))
            result_file = os.path.join(result_dir, result_prefix + '.txt')
            with open(result_file, 'w') as fout:
                fout.write('\n'.join(pred_answers))

        return ave_loss, bleu_rouge

    def save(self, model_dir, model_prefix):
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info("Model save in {}, with prefix {}.".format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info("Model restore from {}, with prefix {}.".format(model_dir, model_prefix))

    def dev_content_answer(self, data_path):
        with open(data_path + ".content", "r") as ref_content_files:
            for content in ref_content_files:
                self.ref_contents.append(content.strip().split())
        with open(data_path + ".answer", "r") as ref_answer_files:
            for answer in ref_answer_files:
                self.ref_answers.append(''.join(answer.strip().split()))


