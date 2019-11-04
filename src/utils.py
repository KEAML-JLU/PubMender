"""
 author: hzhanght
"""
import cPickle as pickle
import numpy as np
import torch


def rec_sents(prob_logits, lengths=None):
    _, idx = torch.max(prob_logits, dim=2)
    idx = idx.cpu().data.tolist()
    if lengths is not None:
        new_idx = []
        for i, l in enumerate(lengths):
            tmp = idx[i][:l-1]
            new_idx.append(tmp)
        idx = new_idx
    return idx


def batch_to_sents(X_batch, lengths):
    sents = []
    for i, l in enumerate(lengths):
        tmp = X_batch[i][:l].tolist()
        sents.append(tmp)
    return sents


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    return zip(range(len(minibatches)), minibatches)


def filter_sentence_by_len(sents, labels=None, maxlen=600, reverse_seq=False):
    lenghts_x = [len(s) for s in sents]
    new_seqs_x = []
    new_labels_x = []
    for i, l in enumerate(lenghts_x):
        if l > maxlen:
            continue
        tmp_sent = sents[i]
        if reverse_seq:
            tmp_sent = np.array(list(reversed(tmp_sent)))
        new_seqs_x.append(tmp_sent)
        if labels is not None:
            new_labels_x.append(labels[i])
    return new_seqs_x, new_labels_x


def split_data_by_label(sents, labels):
    labeled_sents = []
    new_labels = []
    unlabeled_sents = []
    for i, lable in enumerate(labels):
        if lable is None:
            unlabeled_sents.append(sents[i])
        else:
            labeled_sents.append(sents[i])
            new_labels.append(lable)
    return labeled_sents, new_labels, unlabeled_sents


def prepare_data_for_rnn(seqs_x, labels=None, sortL=True):
    lengths_x = [len(s) for s in seqs_x]
    n_samples = len(seqs_x)
    sent_len = max(lengths_x)
    # 0 is Padding token idx
    x = np.zeros(( n_samples, sent_len)).astype('int32')
    for idx, s_x in enumerate(seqs_x):
        x[idx, :lengths_x[idx]] = s_x
    x = torch.from_numpy(x).long()
    if labels is not None:
        labels = torch.LongTensor(labels)
    if sortL:
        x, lengths_x, labels = sort_sents_by_length(x, lengths_x, labels)
    return x, lengths_x, labels


def sort_sents_by_length(sents, lengths, labels=None):
    """
     use the function before load data into gpu
    """
    lengths = torch.LongTensor(lengths)
    lengths, s_idx = torch.sort(lengths, 0, descending=True)
    sents = torch.index_select(sents, 0, s_idx)
    if labels is not None:
        labels = torch.index_select(labels, 0, s_idx)
    return sents, lengths.tolist(), labels


def build_mask(lengths, use_cuda=False, use_heu=False):
    heu = lambda s_tmp: 1.0 + torch.exp(- 100 / torch.arange(1, s_tmp+1))
    max_len = max(lengths)
    batch_size = len(lengths)
    mask = torch.zeros(batch_size, max_len-1)
    for i, l in enumerate(lengths):
        if use_heu:
            mask[i, :l-1] = heu(l-1)
        else:
            mask[i, :l-1] = torch.ones(l-1)
    mask = mask.unsqueeze(2)
    if use_cuda:
        mask = mask.cuda()
    return mask


def convert_sents_to_idx(sents, vocab):
    tmp_sents = []
    for i, sent in enumerate(sents):
        sid = []
        sid.append(vocab.getIdx(vocab.BEGIN_TOKEN))
        for token in sent:
            sid.append(vocab.getIdx(token))
        sid.append(vocab.getIdx(vocab.END_TOKEN))
        tmp_sents.append(np.array(sid))
    return tmp_sents


def dump_preprocessed_data(path, *args):
    with open(path, 'wb') as f:
        pickle.dump(args, f)


def dump_vocab(path, vocab):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)

