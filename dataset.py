
import vocab
class Dataset(object):
    def __init__(self, train_files, dev_files, test_files, max_p_length, max_q_length):
        self.train_set, self.dev_set, self.test_set = [], [], []
        self.max_p_length = max_p_length
        self.max_q_length = max_q_length
        if train_files:
            self.train_set = self.load_dataset(train_files, train=True)
        if dev_files:
            self.train_set = self.load_dataset(dev_files, train=True)
        if train_files:
            self.test_set = self.load_dataset(test_files)

    def load_dataset(self, data_path, train=False):
        dataset = []
        with open(data_path + "content", 'r') as content_files,\
            open(data_path + "question", 'r') as question_files:
            if train:
                span_files = open(data_path + "span", 'r')
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
    def get_batches(self, data_name, batch_size, pad_id, shuffle=True):
        if data_name == 'train':
            data = self.train_set
        elif data_name == 'dev':
            data = self.dev_set
        elif data_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError("No {} data".format(data_name))



