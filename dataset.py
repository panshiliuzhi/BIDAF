
import vocab
import numpy as np
import logging


class Dataset(object):
    def __init__(self, train_files=None, dev_files=None, test_files=None, max_p_length=None, max_q_length=None):
        self.logger = logging.getLogger("BiDAF")
        self.train_set, self.dev_set, self.test_set = [], [], []
        self.max_p_length = max_p_length
        self.max_q_length = max_q_length
        if train_files:
            self.train_set = self.load_dataset(train_files, train=True)
            self.logger.info("Train set size: {} questions".format(len(self.train_set)))
        if dev_files:
            self.dev_set = self.load_dataset(dev_files, train=True)
            self.logger.info("Dev set size: {} questions".format(len(self.dev_set)))
        if test_files:
            self.test_set = self.load_dataset(test_files)
            self.logger.info("Test set size: {} questions".format(len(self.test_set)))

    def load_dataset(self, data_path, train=False):
        dataset = []
        with open(data_path + ".content", 'r') as content_files,\
            open(data_path + ".question", 'r') as question_files:
            if train:
                span_files = open(data_path + ".span", 'r')
                for content, question, span in zip(content_files, question_files, span_files):
                    sample = dict()
                    sample['content'] = content.strip().split()
                    sample['question'] = question.strip().split()
                    sample['span'] = span.strip().split()
                    dataset.append(sample)
            else:
                for content, question in zip(content_files, question_files):
                    sample = dict()
                    sample['content'] = content.strip().split()
                    sample['question'] = question.strip().split()
                    dataset.append(sample)
        return dataset

    def word_iter(self, data_name):
        if data_name == 'train':
            data_set = self.train_set
        elif data_name == 'dev':
            data_set = self.dev_set
        elif data_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No {} data set exists'.format(data_name))
        if data_name is not None:
            for sample in data_set:
                for token in sample['content']:
                    yield token
                for token in sample['question']:
                    yield token

    def convert_to_ids(self, vocab):
        for dataset in [self.train_set, self.dev_set, self.test_set]:
            if dataset is None:
                continue
            for sample in dataset:
                sample['content'] = vocab.convert_to_ids(sample['content'])
                sample['question'] = vocab.convert_to_ids(sample['question'])

    def one_batch(self, data, indices, pad_id):
        batch_data= {
            'question_ids': [],
            'content_ids': [],
            'question_length': [],
            'content_length': [],
            'start': [],
            'end': []
        }
        for i in indices:
            batch_data['question_ids'].append(data[i]['question'][:self.max_q_length])
            batch_data['question_length'].append(min(len(data[i]['question']), self.max_q_length))
            batch_data['content_ids'].append(data[i]['content'][:self.max_p_length])
            batch_data['content_length'].append(min(len(data[i]['content']), self.max_p_length))
            if 'span' in data[i]:
                batch_data['start'].append(int(data[i]['span'][0]))
                batch_data['end'].append(int(data[i]['span'][1]))
        return self.padding(batch_data, pad_id)

    def padding(self, batch_data, pad_id):
        pad_p_length = min(self.max_p_length, max(batch_data['content_length']))
        pad_q_length = min(self.max_q_length, max(batch_data['question_length']))

        batch_data['content_ids'] = [(ids +[pad_id]*(pad_p_length - len(ids)))[:pad_p_length]
                                     for ids in batch_data['content_ids']]
        batch_data['question_ids'] = [(ids + [pad_id]*(pad_q_length - len(ids)))[:pad_q_length]
                                      for ids in batch_data['question_ids']]
        return batch_data

    def get_batches(self, data_name, batch_size, pad_id, shuffle=True):
        if data_name == 'train':
            data = self.train_set
        elif data_name == 'dev':
            data = self.dev_set
        elif data_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError("No {} data".format(data_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for start_index in np.arange(0, data_size, batch_size):
            batch_indices = indices[start_index : start_index + batch_size]
            yield self.one_batch(data, batch_indices, pad_id)

