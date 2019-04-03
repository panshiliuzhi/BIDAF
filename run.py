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
from RNETModel import RNETModel
from QANet import QANetModel
import jieba
import numpy as np
import tensorflow as tf
from model_multi_gpu import training
from bilm import TokenBatcher

from layers.params import Params
def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('Reading Comprehension on Les MMRC dataset')
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
    parser.add_argument('--gpu', type=str, default='1',
                        help='specify gpu device')
    parser.add_argument('--use_embe', type=int, default=1, help='is use embeddings vector file')

    train_settings = parser.add_argument_group('train settings')
    train_settings.add_argument('--learning_rate', type=float, default=0.001,
                                help='learning rate')
    train_settings.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')
    train_settings.add_argument('--dropout_keep_prob', type=float, default=0.8,
                                help='dropout keep rate')
    train_settings.add_argument('--batch_size', type=int, default=64,
                                help='train batch size')
    train_settings.add_argument('--epochs', type=int, default=100,
                                help='train epochs')

    model_settings = parser.add_argument_group('model settings')

    model_settings.add_argument('--algo', type=str, choices=["BIDAF-elmo", "R-net-elmo", "R-net-elmo-1", "QANet"], default='BIDAF-elmo',
                                help='choose the prefix to save')
    model_settings.add_argument('--embed_size', type=int, default=300,
                                help='size of the embeddings')
    model_settings.add_argument('--hidden_size', type=int, default=75,
                                help='size of LSTM hidden units')
    model_settings.add_argument('--head_count', type=int, default=4,
                                help='the number of attention head')
    model_settings.add_argument('--max_p_len', type=int, default=400,
                                help='max length of passage')
    model_settings.add_argument('--max_q_len', type=int, default=80,
                                help='max length of question')
    model_settings.add_argument('--max_a_len', type=int, default=40,
                                help='max length of answer')
    model_settings.add_argument('--n_gpus', type=int, default=3,
                                help='the number of gpu')

    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--train_files', default='/home/home1/dmyan/codes/BIDAF/data/3/train',
                               help='the file of train data')
    path_settings.add_argument('--dev_files', default='/home/home1/dmyan/codes/BIDAF/data/3/dev',
                               help='the file of dev data')
    path_settings.add_argument('--test_files', default='/home/home1/dmyan/codes/BIDAF/data/3/test',
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
    logger = logging.getLogger(args.algo)
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
        vocab.load_pretrained_embeddings(embedding_path='/home/home1/dmyan/codes/tensorflow/data/word2vec/300_ver_not_pure.bin')
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
    logger = logging.getLogger(args.algo)
    logger.info('Load data_set and vocab...')
    # with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
    #     vocab = pickle.load(fin)

    data_dir = '/home/home1/dmyan/codes/bilm-tf/bilm/data/'
    vocab_file = data_dir + 'vocab.txt'
    batcher = TokenBatcher(vocab_file)

    data = Dataset(train_files=args.train_files, dev_files=args.dev_files, max_p_length=args.max_p_len, max_q_length=args.max_q_len)
    logger.info('Converting text into ids...')
    data.convert_to_ids(batcher)
    logger.info('Initialize the model...')
    if args.algo.startswith("BIDAF"):
        model = BiDAFModel(args)
    elif args.algo.startswith("R-net"):
        model = RNETModel(args)
    elif args.algo.startswith("QANET"):
        model = QANetModel(args)
    #model.restore(model_dir=args.model_dir, model_prefix=args.algo)
    logger.info("Load dev dataset...")
    model.dev_content_answer(args.dev_files)
    logger.info('Training the model...')
    model.train(data, args.epochs, args.batch_size, save_dir=args.model_dir+args.algo,
                   save_prefix=args.algo,
                   dropout_keep_prob=args.dropout_keep_prob)
    logger.info('Done with model training!')


def multi_gpu_train(args):
    """
    multi gpus train the reading comprehension model
    """
    logger = logging.getLogger("BiDAF")
    logger.info('Load data_set and vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
        vocab = pickle.load(fin)

    data = Dataset(train_files=args.train_files, dev_files=args.dev_files, max_p_length=args.max_p_len,
                   max_q_length=args.max_q_len)
    logger.info('Converting text into ids...')
    data.convert_to_ids(vocab)
    training(args, data, vocab)





def evaluate(args):
    """
    evaluate the trained model on dev files
    """

    logger = logging.getLogger(args.algo)
    logger.info('Load data_set and vocab...')
    # with open(os.path.join(args.vocab_dir, 'vocab.data'), 'rb') as fin:
    #     vocab = pickle.load(fin)

    data_dir = '/home/home1/dmyan/codes/bilm-tf/bilm/data/'
    vocab_file = data_dir + 'vocab.txt'
    batcher = TokenBatcher(vocab_file)

    data = Dataset(test_files=args.test_files, max_p_length=args.max_p_len, max_q_length=args.max_q_len)
    logger.info('Converting text into ids...')
    data.convert_to_ids(batcher)
    logger.info('Initialize the model...')
    if args.algo.startswith("BIDAF"):
        model = BiDAFModel(args)
    elif args.algo.startswith("R-net"):
        model = RNETModel(args)
    model.restore(model_dir=args.model_dir+args.algo, model_prefix=args.algo)
    #logger.info("Load dev dataset...")
    #model.dev_content_answer(args.dev_files)
    logger.info('Testing the model...')
    eval_batches = data.get_batches("test", args.batch_size, 0, shuffle=False)
    eval_loss, bleu_rouge = model.evaluate(eval_batches, result_dir=args.result_dir, result_prefix="test.predicted")
    logger.info("Test loss {}".format(eval_loss))
    logger.info("Test result: {}".format(bleu_rouge))
    logger.info('Done with model Testing!')

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

    logger = logging.getLogger(args.algo)
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

    Params.hidden_size = args.hidden_size
    Params.batch_size = args.batch_size
    Params.max_q_len = args.max_q_len

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.evaluate:
        evaluate(args)
    if args.predict:
        predict(args)
    if args.multi_gpu_train:
        multi_gpu_train(args)
    if args.interactive:
        interactive(args)

if __name__ == '__main__':
    run()
