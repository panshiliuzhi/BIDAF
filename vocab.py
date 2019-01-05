import gzip
import numpy as np
class Vocab(object):

    def __init__(self):
        self.id2token = {}
        self.token2id = {}
        self.token_cnt = {}
        self.embeddings = None
        self.embed_dim = None
        self.pad_token = "pad"
        self.unk_token = "unk"
        self.add(self.pad_token)
        self.add(self.unk_token)


    def size(self):
        return len(self.id2token)

    def add(self, token, cnt=1):
        if token not in self.token2id:
            idx = len(self.id2token)
            self.token2id[token] = idx
            self.id2token[idx] = token
        if cnt > 0:
            if token in self.token_cnt:
                self.token_cnt[token] += cnt
            else:
                self.token_cnt[token] = cnt

    def filtered_tokens(self, min_cnt):
        filtered_tokens = [token for token in self.token2id if self.token_cnt[token] >= min_cnt]
        self.token2id = {}
        self.id2token = {}
        self.add(self.pad_token, cnt=0)
        self.add(self.unk_token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)


    def get_id(self, token):
        try:
            return self.token2id[token]
        except KeyError:
            return self.token2id[self.unk_token]

    def convert_to_ids(self, tokens):
        vec = [self.get_id(token) for token in tokens]
        return vec

    def load_pretrained_embeddings(self, embedding_path):
        trained_embeddings = {}
        with gzip.open(embedding_path, 'rt', encoding='utf-8') as fin:
            for embed in fin:
                embed = embed.strip().split()
                word = embed[0]
                if word not in self.token2id:
                    continue
                trained_embeddings[word] = list(map(float, embed[1:]))
                if self.embed_dim is None:
                    self.embed_dim = len(embed) - 1
        filtered_tokens = trained_embeddings.keys()
        # rebuild the token x id map
        self.token2id = {}
        self.id2token = {}
        self.add(self.pad_token, cnt=0)
        self.add(self.unk_token, cnt=0)
        for token in filtered_tokens:
            self.add(token, cnt=0)
        self.embeddings = np.zeros(self.size(), self.embed_dim)
        for token in self.token2id.keys():
            if token in trained_embeddings:
                self.embeddings[self.get_id(token)] = trained_embeddings[token]

    def random_init_embeddings(self, embed_dim):
        self.embed_dim = embed_dim
        self.embeddings = np.random.rand(self.size(), embed_dim)
        for token in [self.pad_token, self.unk_token]:
            self.embeddings[self.get_id(token)] = np.zeros(embed_dim)

