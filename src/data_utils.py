"""
 author: hzhanght
"""
import os
import io
import nltk
import numpy as np


class Vocab(object):
    PAD_TOKEN = 'H_Z_PAD'
    BEGIN_TOKEN = 'H_Z_BEGIN'
    END_TOKEN = 'H_Z_END'
    UNK_TOKEN = 'H_Z_UNK'

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = {}
        self.special_tokens = [self.PAD_TOKEN, self.BEGIN_TOKEN, self.END_TOKEN, self.UNK_TOKEN]
        for token in self.special_tokens:
            self.add_word(token)
        assert self.word2idx[self.PAD_TOKEN] == 0
        assert self.word2idx[self.BEGIN_TOKEN] == 1
        assert self.word2idx[self.END_TOKEN] == 2
        assert self.word2idx[self.UNK_TOKEN] == 3

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.idx2word)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.word_count[word] = 1
        else:
            self.word_count[word] += 1

    @property
    def vocab_len(self):
        return len(self.idx2word)

    def __len__(self):
        return self.vocab_len

    def shrink_vocab(self, threshold=1):
        word2idx = {}
        idx2word = {}
        word_count = {}
        for wd, c in self.word_count.items():
            if c >= threshold:
                word_count[wd] = c
        for token in self.special_tokens:
            word_count[token] = 1
            word2idx[token] = len(word2idx)
            idx2word[word2idx[token]] = token
        for wd in word_count.keys():
            if wd in self.special_tokens:
                continue
            word2idx[wd] = len(word2idx)
            idx2word[word2idx[wd]] = wd
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.word_count = word_count
        assert len(word2idx) == len(word_count)
        assert len(word2idx) == len(idx2word)
        assert self.word2idx[self.PAD_TOKEN] == 0, 'pad {}'.format(self.word2idx[self.PAD_TOKEN])
        assert self.word2idx[self.BEGIN_TOKEN] == 1, 'begin {}'.format(self.word2idx[self.BEGIN_TOKEN])
        assert self.word2idx[self.END_TOKEN] == 2, 'end {}'.format(self.word2idx[self.END_TOKEN])
        assert self.word2idx[self.UNK_TOKEN] == 3, 'unk {}'.format(self.word2idx[self.UNK_TOKEN])

    def shrink_vocab_to_count(self, count=5000):
        if count < 1:
            count = 5000
        tmp_word_count = self.word_count.items()
        tmp_word_count = sorted(tmp_word_count, key=lambda x: x[1], reverse=True)
        tmp_word_count = tmp_word_count[:count]
        word_count = dict(tmp_word_count)
        word2idx = {}
        idx2word = {}
        for token in self.special_tokens:
            word_count[token] = 1
            word2idx[token] = len(word2idx)
            idx2word[word2idx[token]] = token
        for wd in word_count.keys():
            if wd in self.special_tokens:
                continue
            word2idx[wd] = len(word2idx)
            idx2word[word2idx[wd]] = wd
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.word_count = word_count
        assert len(word2idx) == len(word_count)
        assert len(word2idx) == len(idx2word)
        assert self.word2idx[self.PAD_TOKEN] == 0, 'pad {}'.format(self.word2idx[self.PAD_TOKEN])
        assert self.word2idx[self.BEGIN_TOKEN] == 1, 'begin {}'.format(self.word2idx[self.BEGIN_TOKEN])
        assert self.word2idx[self.END_TOKEN] == 2, 'end {}'.format(self.word2idx[self.END_TOKEN])
        assert self.word2idx[self.UNK_TOKEN] == 3, 'unk {}'.format(self.word2idx[self.UNK_TOKEN])


    def getIdx(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx[self.UNK_TOKEN]

    def getWord(self, idx):
        if idx in self.idx2word:
            return self.idx2word[idx]
        else:
            return self.UNK_TOKEN

    def isExist(self, word):
        if word in self.word2idx:
            return True
        else:
            return False

    @staticmethod
    def build_vocab(sents, fixed_vocab_set=None):
        vocab = Vocab()
        if False:
            for sent in sents:
                for token in sent:
                    if fixed_vocab_set and token not in fixed_vocab_set:
                        continue
                    vocab.add_word(token)
        else:
            vocab.append_sents(sents, fixed_vocab_set)
        return vocab

    def append_sents(self, sents, fixed_vocab_set=None):
        for sent in sents:
            for token in sent:
                if fixed_vocab_set and token not in fixed_vocab_set:
                    continue
                self.add_word(token)


class PubMedReader(object):

    def __init__(self, opt):
        self.data_path = opt.data_path
        self.data_split = ['train', 'test', 'valid']
        self.label2idx = opt.label2idx

    def read_pubmed_subdir(self, subdir='train'):
        all_lables = []
        all_sents = []
        subdir = os.path.join(self.data_path, subdir)
        for journal_name in self.label2idx.keys():
            journal_path = os.path.join(subdir, journal_name)
            jidx = self.label2idx[journal_name]
            assert os.path.isfile(journal_path), "{} {}".format(journal_path, journal_name)
            with io.open(journal_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip()
                    all_sents.append(line)
                    all_lables.append(jidx)
        return all_sents, all_lables

    def get_data(self, shuffle=True):
        tmp = []
        for s in self.data_split:
            sents, lables = self.read_pubmed_subdir(s)
            sents = [self.process_sent(sent) for sent in sents]
            if shuffle:
                sents = np.array(sents)
                lables = np.array(lables)
                p_idx = np.random.permutation(len(sents))
                sents = sents[p_idx]
                lables = lables[p_idx]
                sents = sents.tolist()
                lables = lables.tolist()
            tmp.append(sents)
            tmp.append(lables)
        return tmp

    @staticmethod
    def process_sent(sent):
        tokens = nltk.word_tokenize(sent)
        return tokens


if __name__ == '__main__':
    import config
    opt = config.Options(config_vocab=False)
