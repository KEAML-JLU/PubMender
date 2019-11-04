"""
 author:hzhanght
"""
from data_utils import Vocab, PubMedReader
from utils import convert_sents_to_idx, dump_preprocessed_data, dump_vocab
from config import Options
import io
import numpy as np
from models import Embedding
import torch

def read_vocab(w2v_path):
    vocab = []
    with io.open(w2v_path) as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            w, _ = l.split(' ',1)
            vocab.append(w)
    return set(vocab)

def read_vec(w2v_path, w_list):
    vec_dic = {}
    w_set = set(w_list)
    vec_dim = None
    with io.open(w2v_path) as f:
        for l in f:
            l = l.strip()
            if not l:
                continue
            w, v = l.split(' ',1)
            if w in w_set:
                v = np.array(map(float, v.split()))
                vec_dic[w] = v
                if vec_dim is None:
                    vec_dim = len(v)
    vec_list = []
    for i, w in enumerate(w_list):
        if w in vec_dic:
            vec_list.append(vec_dic[w])
        else:
            print(i, w)
            vec_list.append(np.random.rand(200))
    return np.array(vec_list)

if __name__ == '__main__':
    vocab_num = 100000
    pubmed_w2v_path = 'pubmed_w2v.txt'
    emb_path = 'emb_cnn.pt'
    opt = Options(config_vocab=False)
    pubmedreader = PubMedReader(opt)
    print('loding text data')
    train_sents, train_labels, test_sents, test_labels, valid_sents, valid_labels = pubmedreader.get_data()

    print('read vocab')
    fixed_vocab_set = read_vocab(pubmed_w2v_path)
    print('fixed vocab set size {}'.format(len(fixed_vocab_set)))
    print('build vocab')
    vocab = Vocab.build_vocab(train_sents, fixed_vocab_set=fixed_vocab_set)
    #
    vocab.append_sents(valid_sents, fixed_vocab_set=fixed_vocab_set)
    vocab.append_sents(test_sents, fixed_vocab_set=fixed_vocab_set)
    # 
    print('vocab size {} before shrink'.format(vocab.vocab_len))
    vocab.shrink_vocab(2)
    print('vocab size {} after shrink'.format(vocab.vocab_len))

    print('read vec')
    word_list = [vocab.idx2word[i] for i in range(len(vocab.idx2word))]
    vec = read_vec(pubmed_w2v_path, word_list)
    assert vec.shape[0] == vocab.vocab_len

    print('build emb layer')
    emb = Embedding(vocab.vocab_len, vec.shape[1], padding_idx=0, trainable=False)
    emb.initialize_embedding(vec)
    emb.cuda()
    torch.save(emb.state_dict(), emb_path)

    print('dump data')
    train_sents = convert_sents_to_idx(train_sents, vocab)
    test_sents = convert_sents_to_idx(test_sents, vocab)
    valid_sents = convert_sents_to_idx(valid_sents, vocab)
    dump_preprocessed_data(opt.train_path, train_sents, train_labels)
    dump_preprocessed_data(opt.test_path, test_sents, test_labels)
    dump_preprocessed_data(opt.valid_path, valid_sents, valid_labels)
    dump_vocab(opt.vocab_path, vocab)
