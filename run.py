# -*- coding:utf8 -*-
"""
This module prepares and runs the whole system.
"""
import sys
# if sys.version[0] == '2':
#     reload(sys)
#     sys.setdefaultencoding("utf-8")
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
import argparse
import logging
from dataset import Dataset
from vocab import Vocab
from model import BiDAFModel
import jieba
import numpy as np
import tensorflow as tf


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on RC dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--interactive', action='store_true',
                        help='interact with user')
    parser.add_argument('--multi_gpu_train', action='store_true',
                        help='train the model on multi gpu')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model on dev set')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')
    parser.add_argument('--use_embe', type=int, default=1, help='is use embeddings vector file')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--learning_rate', type=float, default=0.0001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=0.5,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=4,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=50,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=150,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--max_p_len', type=int, default=800,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=80,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=40,
                                help='max length of answer')
    model_settings.add_argument('--n_gpus', type=int, default=1,
                                help='the number of gpu')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', default='/home/dmyan/home/BIDAF/data/debug/train',
                               help='the file of train data')
    path_settings.add_argument('--dev_files', default='/home/dmyan/home/BIDAF/data/debug/dev',
                               help='the file of dev data')
    path_settings.add_argument('--test_files', default='/home/dmyan/home/BIDAF/data/debug/test',
                               help='the file of test data')
    path_settings.add_argument('--vocab_dir', default='./data/vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--model_dir', default='./data/models/',
                               help='the dir to store models')
    path_settings.add_argument('--result_dir', default='./data/results/',
                               help='the dir to output the results')
    path_settings.add_argument('--summary_dir', default='./data/summary/',
                               help='the dir to write tensorboard summary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()


def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("BiDAF")
    logger.info('Preparing the directories...')
    for dir_path in [args.vocab_dir, args.model_dir, args.result_dir, args.summary_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    logger.info('Building vocabulary...')
    data = Dataset(args.train_files, args.dev_files, args.test_files
                        , args.max_p_len, args.max_q_len)
    vocab = Vocab()
    for word in data.word_iter('train'):
        vocab.add(word)
    unfiltered_vocab_size = vocab.size()
    vocab.filtered_tokens(min_cnt=2)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))
    logger.info('Assigning embeddings...')
    if args.use_embe:
        vocab.load_pretrained_embeddings(embedding_path='/home/dmyan/home/tensorflow/data/word2vec/300_ver_not_pure.bin')
    else:
        vocab.random_init_embeddings(args.embed_size)

    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout)

    logger.info('Done with preparing!')


def train(args):
    """
    trains the reading comprehension model
    """
    logger = logging.getLogger("BiDAF")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)

    data = Dataset(train_files=args.train_files, dev_files=args.dev_files, max_p_length=args.max_p_len, max_q_length=args.max_q_len)
    logger.info('Converting text into ids...')
    data.convert_to_ids(vocab)
    logger.info('Initialize the model...')
    model = BiDAFModel(vocab, args)
    #model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    #logger.info("Load dev dataset...")
    #rc_model.dev_content_answer(args.dev_files)
    logger.info('Training the model...')
    model.train(data, args.epochs, save_dir=args.model_dir,
                   save_prefix="BIDAF",
                   dropout_keep_prob=args.dropout_keep_prob)
    logger.info('Done with model training!')


# def multi_gpu_train(args):
#     """
#     multi gpus train the reading comprehension model
#     """
#     logger = logging.getLogger("mc")
#     logger.info('Load data_set and vocab...')
#     with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
#         vocab = pickle.load(fin)
#
#     brc_data = BRCDataset( args.max_p_len, args.max_q_len,
#                           args.train_files, args.dev_files)
#     logger.info('Converting text into ids...')
#     brc_data.convert_to_ids(vocab)
#     logger.info('Initialize the model...')
#     gpu_avaiables = [0, 2]
#     batch_size = args.batch_size
#     with tf.device("/cpu:0"):
#         tower_grads = []
#         p = tf.placeholder(tf.int32, [None, None])
#         q = tf.placeholder(tf.int32, [None, None])
#         p_length = tf.placeholder(tf.int32, [None])
#         q_length = tf.placeholder(tf.int32, [None])
#         start_label = tf.placeholder(tf.int32, [None])
#         end_label = tf.placeholder(tf.int32, [None])
#         dropout_keep_prob = tf.placeholder(tf.float32)
#         with tf.variable_scope('word_embedding'):
#             word_embeddings = tf.get_variable(
#                 'word_embeddings',
#                 shape=(vocab.size(), vocab.embed_dim),
#                 initializer=tf.constant_initializer(vocab.embeddings),
#                 trainable=True
#             )
#             p_emb = tf.nn.embedding_lookup(word_embeddings, p)
#             q_emb = tf.nn.embedding_lookup(word_embeddings, q)
#         opt = tf.train.AdadeltaOptimizer(args.learning_rate)
#         sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
#         sess_config.gpu_options.allow_growth = True
#
#         #sess = tf.Session(config=sess_config)
#         with tf.Session(config=sess_config) as sess:
#             with tf.variable_scope(tf.get_variable_scope()):
#                 for i in range(args.n_gpus):
#                     with tf.device("/gpu:%d" % gpu_avaiables[i]):
#                         with tf.name_scope("tower_%d" % gpu_avaiables[i]):
#                             _p = p[i * batch_size:(i + 1) * batch_size]
#                             _q = q[i * batch_size:(i + 1) * batch_size]
#                             _p_length = p_length[i * batch_size:(i + 1) * batch_size]
#                             _q_length = q_length[i * batch_size:(i + 1) * batch_size]
#
#                             _start_label = start_label[i * batch_size:(i + 1) * batch_size]
#                             _end_label = end_label[i * batch_size:(i + 1) * batch_size]
#                             _p_emb = p_emb[i * batch_size:(i + 1) * batch_size]
#                             _q_emb = q_emb[i * batch_size:(i + 1) * batch_size]
#                             data = (_p, _q, _p_length, _q_length, _start_label, _end_label, _p_emb, _q_emb, dropout_keep_prob)
#                             model = RCModel_ngpus(args, data)
#                             tf.get_variable_scope().reuse_variables()
#                             model_loss = model.loss
#                             grads = opt.compute_gradients(model_loss)
#                             tower_grads.append(grads)
#             grads = average_gradients(tower_grads)
#             train_op =  opt.apply_gradients(grads)
#
#             sess.run(tf.global_variables_initializer())
#             pad_id = vocab.get_id(vocab.pad_token)
#             for epoch in range(1, args.epochs+1):
#                 train_batches = brc_data.gen_mini_batches('train', batch_size*args.n_gpus, pad_id, shuffle=True)
#                 total_num, total_loss = 0, 0
#                 log_every_n_batch, n_batch_loss = 50, 0
#
#                 for bitx, batch in enumerate(train_batches, 1):
#                     feed_dict = {p: batch['content_ids'],
#                                  q: batch['question_ids'],
#                                  p_length: batch['content_length'],
#                                  q_length: batch['question_length'],
#                                  start_label: batch['start_id'],
#                                  end_label: batch['end_id'],
#                                  dropout_keep_prob: dropout_keep_prob}
#                     _, loss = sess.run([train_op, model_loss], feed_dict)
#
#                     total_loss += loss
#                     total_num += 1
#                     n_batch_loss += loss
#                     if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
#                         logger.info('Average loss from batch {} to {} is {}'.format(
#                             bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
#                         n_batch_loss = 0
#                 train_loss = 1.0 * total_loss / total_num
#                 logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

def evaluate(args):
    """
    evaluate the trained model on dev files
    """
    logger = logging.getLogger("BiDAF")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.dev_files) > 0, 'No dev files are provided.'
    brc_data = Dataset(args.max_p_num, args.max_p_len, args.max_q_len, dev_files=args.dev_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = BiDAFModel(vocab, args)

    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Evaluating the model on dev set...')
    dev_batches = brc_data.gen_mini_batches('dev', args.batch_size,
                                            pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    dev_loss, dev_bleu_rouge = rc_model.evaluate(
        dev_batches, result_dir=args.result_dir, result_prefix='dev.predicted')
    logger.info('Loss on dev set: {}'.format(dev_loss))
    logger.info('Result on dev set: {}'.format(dev_bleu_rouge))
    logger.info('Predicted answers are saved to {}'.format(os.path.join(args.result_dir)))

def interactive(args):
    logger = logging.getLogger("BiDAF")
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)

    logger.info('Restoring the model...')
    rc_model = BiDAFModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)


    while True:
        content = input('输入原文:\n')
        if content == 'exit' :
            exit(0)

        question = input('\n输入问题:\n')
        if  question == 'exit':
            exit(0)

        content_segs = ' '.join(jieba.cut(content)).split()
        question_segs = ' '.join(jieba.cut(question)).split()

        content_ids = vocab.convert_to_ids(content_segs)[:args.max_p_len]
        question_ids = vocab.convert_to_ids(question_segs)[:args.max_q_len]

        batch_data = {'question_ids': [],
                      'question_length': [],
                      'content_ids': [],
                      'content_length': [],
                      'start_id': [],
                      'end_id': []}
        batch_data['question_ids'].append(question_ids)
        batch_data['question_length'].append(len(question_ids))
        batch_data['content_ids'].append(content_ids)
        batch_data['content_length'].append(len(content_ids))
        batch_data['start_id'].append(0)
        batch_data['end_id'].append(0)

        start_idx, end_idx = rc_model.getAnswer(batch_data)
        print('\n==========================================================================\n')
        print('答案是 :', ''.join(content_segs[start_idx:end_idx+1]))
        print('\n==========================================================================\n')

def predict(args):
    """
    predicts answers for test files
    """
    logger = logging.getLogger("BiDAF")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)
    assert len(args.test_files) > 0, 'No test files are provided.'
    brc_data = Dataset( args.max_p_len, args.max_q_len,
                          test_files=args.test_files)
    logger.info('Converting text into ids...')
    brc_data.convert_to_ids(vocab)
    logger.info('Restoring the model...')
    rc_model = BiDAFModel(vocab, args)
    rc_model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info('Predicting answers for test set...')
    test_batches = brc_data.gen_mini_batches('test', args.batch_size,
                                             pad_id=vocab.get_id(vocab.pad_token), shuffle=False)
    rc_model.evaluate(test_batches,
                      result_dir=args.result_dir, result_prefix='test.predicted')


def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger("BiDAF")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler( )
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)
    if args.interactive:
        interactive(args)

if __name__ == '__main__':
    run()
