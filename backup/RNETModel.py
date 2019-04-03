import logging
import tensorflow as tf
import os
import time
import numpy as np
import math
from evaluate.bleu import Bleu
from evaluate.rouge import RougeL
from layers.rnn import rnn, gatedAttentionGRU, get_attn_params
from layers.rnn import linear
from layers.output_linear import answer_pointer
from bilm import  BidirectionalLanguageModel, weight_layers



class RNETModel(object):
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
        self.params = get_attn_params(self.hidden_size, tf.contrib.layers.xavier_initializer)
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
        self.encode()
        self.multiHead_attention()
        self.output_layer()
        self.compute_loss()
        self.logger.info("Time to build graph: {}s".format(time.time() - start_time))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info("The total parmas:{}".format(param_num))

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
            self.u_p = rnn('gru', self.p_embed, self.hidden_size, self.p_length, layer_num=1)
        with tf.variable_scope("question_encoding"):
            self.u_q = rnn('gru', self.q_embed, self.hidden_size, self.q_length, layer_num=1)
        if self.use_dropout:
            self.u_p = tf.nn.dropout(self.u_p, self.dropout_keep_prob)
            self.u_q = tf.nn.dropout(self.u_q, self.dropout_keep_prob)

    def multiHead_attention(self):
        params = [(([self.params["w_u_q"], self.params["w_u_p"], self.params["w_v_p"]], self.params["v"]),self.params["w_g"]),
                  (([self.params["w_v_p_"], self.params["w_v_phat"]], self.params["v"]), self.params["w_g_"])]
        head_output = []
        with tf.variable_scope("multiHead_attention"):
            for i in range(self.head_count):
                q = linear(tf.reshape(self.u_q, (-1, 2*self.hidden_size)), 2*self.hidden_size, bias=False, scope='headq'+str(i))
                p = linear(tf.reshape(self.u_p, (-1, 2*self.hidden_size)), 2*self.hidden_size, bias=False, scope='headp'+str(i))
                q = tf.reshape(q, (self.batch_size, -1, 2*self.hidden_size))
                p = tf.reshape(p, (self.batch_size, -1, 2*self.hidden_size))
                with tf.variable_scope("head"+str(i)):
                    head = gatedAttentionGRU(q, self.q_length, p, self.p_length, self.hidden_size, params[0], use_state=True, dropout_keep_prob=self.dropout_keep_prob)
                head_output.append(head)
            head_output = tf.concat(head_output, -1)
            multi_head = linear(tf.reshape(head_output, (-1, self.head_count*2*self.hidden_size)), 2*self.hidden_size, False, scope='multi_head')
            multi_head = tf.maximum(
                             linear(multi_head, self.hidden_size*2, bias=True, bias_start=1.0, scope="transformer1"), 0)
            multi_head = tf.nn.relu(linear(multi_head, self.hidden_size*2, bias=True, bias_start=1.0, scope="transformer2"))
            self.v_p = self.u_p + tf.reshape(multi_head, (self.batch_size, -1, 2*self.hidden_size))
        with tf.variable_scope("self_matching"):
            self.h_p = self.v_p + gatedAttentionGRU(self.v_p, self.p_length, self.v_p, self.p_length, self.hidden_size, params[1], dropout_keep_prob=self.dropout_keep_prob)


    # def gated_attention(self):
    #     params = [(([self.params["w_u_q"], self.params["w_u_p"], self.params["w_v_p"]], self.params["v"]),self.params["w_g"]),
    #               (([self.params["w_v_p_"], self.params["w_v_phat"]], self.params["v"]), self.params["w_g_"])]
    #     with tf.variable_scope("attention_pooling"):
    #         self.v_p = self.u_p + gatedAttentionGRU(self.u_q, self.q_length, self.u_p, self.p_length, self.hidden_size, params[0], use_state=True, dropout_keep_prob=self.dropout_keep_prob)
    #
    #     with tf.variable_scope("self_matching"):
    #         self.h_p = self.v_p + gatedAttentionGRU(self.v_p, self.p_length, self.v_p, self.p_length, self.hidden_size, params[1], dropout_keep_prob=self.dropout_keep_prob)

    def output_layer(self):
        params = [
            ([self.params["w_h_p"], self.params["w_h_a"]], self.params["v"]),
            ([self.params["w_u_q_"], self.params["w_v_q"]], self.params["v"])
        ]
        with tf.variable_scope("output_layer"):
            self.h_p = rnn("gru", self.h_p, self.hidden_size, self.p_length, dropout_keep_prob=self.dropout_keep_prob)
        self.p1, self.p2 = answer_pointer(self.h_p, self.p_length, self.u_q, self.q_length, self.hidden_size, params, self.batch_size)
    def compute_loss(self):
        def log_loss(probs, y, epsion=1e-9):
            with tf.name_scope("log_loss"):
                y = tf.one_hot(y, tf.shape(probs)[1], axis=1)
                loss = - tf.reduce_sum(y * tf.log(probs + epsion), 1)
                return loss
        self.start_loss = log_loss(self.p1, self.start)
        self.end_loss = log_loss(self.p2, self.end)
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        self.train_op = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate, rho=0.95, epsilon=1e-6).minimize(self.loss)
        #self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def train_one_epoch(self, batch_datas, dropout_keep_prob):
        total_loss , total_num = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        for bitx, batch in enumerate(batch_datas, 1):
            feed_dict = {
                self.W_P:batch["content_ids"],
                self.W_Q:batch["question_ids"],
                self.p_length:batch["content_length"],
                self.q_length:batch["question_length"],
                self.start:batch["start"],
                self.end:batch["end"],
                self.dropout_keep_prob:dropout_keep_prob
            }
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            total_loss += loss
            total_num += 1
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info("Average loss from batch {} to {} is {}".format(
                    bitx - log_every_n_batch +1, bitx, n_batch_loss/log_every_n_batch
                ))
                n_batch_loss = 0
        return 1.0*total_loss/total_num


    def train(self, data, epochs, batch_size, save_dir=None, save_prefix=None, dropout_keep_prob=1.0, evaluate=True):
        pad_id = 0
        max_rougeL = 0
        for epoch in range(1, epochs+1):
            self.logger.info("Training the model for epoch {}".format(epoch))
            batch_datas = data.get_batches("train", batch_size, pad_id, shuffle=True)
            train_loss = self.train_one_epoch(batch_datas, dropout_keep_prob)
            self.logger.info("Average train loss for epoch {} is {}".format(epoch, train_loss))


            if evaluate:
                self.logger.info("Evaluating the model after epoch {}".format(epoch))
                eval_batches = data.get_batches("dev",batch_size, pad_id, shuffle=False)
                eval_loss, bleu_rouge = self.evaluate(eval_batches)
                self.logger.info("Dev evaluate loss {}".format(eval_loss))
                self.logger.info("Dev evaluate result: {}".format(bleu_rouge))
                if math.isnan(bleu_rouge['Bleu-4']) or math.isnan(bleu_rouge['Rouge-l']):
                    self.logger.info("Dev eval is nan!")
                    continue
                if bleu_rouge['Rouge-l'] > max_rougeL:
                    self.save(save_dir, save_prefix)
                    max_rougeL = bleu_rouge['Rouge-l']
            else:
                self.logger.warning('No dev set is loaded for evaluation in the dataset!')



    def evaluate(self, eval_batches, result_dir=None, result_prefix=None):
        total_loss, total_num = 0, 0
        start_indices = []
        end_indices = []
        for bitx, batch in enumerate(eval_batches, 1):
            feed_dict = {
                self.W_P: batch["content_ids"],
                self.W_Q: batch["question_ids"],
                self.p_length: batch["content_length"],
                self.q_length: batch["question_length"],
                self.start: batch["start"],
                self.end: batch["end"],
                self.dropout_keep_prob: 1.0
            }
            start_probs, end_probs, loss = self.sess.run([self.p1, self.p2, self.loss], feed_dict=feed_dict)
            start_indices += np.argmax(start_probs, axis=1).tolist()
            end_indices += np.argmax(end_probs, axis=1).tolist()
            total_loss += loss
            total_num += 1
        rouge_eval = RougeL()
        blue_eval = Bleu()
        if result_dir is not None and result_prefix is not None:
            with open("/home/home1/dmyan/codes/tensorflow/data/test.answer", "r") as ref_answer_files:
                for answer in ref_answer_files:
                    self.ref_answers.append(''.join(answer.strip().split()))
            with open("/home/home1/dmyan/codes/tensorflow/data/test.content", "r") as ref_content_files:
                for content in ref_content_files:
                    self.ref_contents.append(content.strip().split())
        pred_answers = []
        for i, (start, end) in enumerate(zip(start_indices, end_indices)):
            end  = np.where(end<start, start+self.max_a_length, min(end, start+self.max_a_length))
            answer = ''.join(self.ref_contents[i][start:end+1])
            if result_prefix is not None and result_prefix is not None:
                pred_answers.append(answer)
            rouge_eval.add_inst(answer, self.ref_answers[i])
            blue_eval.add_inst(answer, self.ref_answers[i])
        bleu_score = blue_eval.get_score()
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


    def dev_content_answer(self, data_path):
        with open(data_path+".content", "r") as ref_content_files:
            for content in ref_content_files:
                self.ref_contents.append(content.strip().split())
        with open(data_path+".answer", "r") as ref_answer_files:
            for answer in ref_answer_files:
                self.ref_answers.append(''.join(answer.strip().split()))


    def save(self, model_dir, model_prefix):
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info("Model save in {} with prefix {}".format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info("Model restore from {} with prefix{}".format(model_dir, model_prefix))



