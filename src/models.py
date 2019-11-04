"""
 author:hzhanght
"""
from __future__ import print_function, absolute_import, division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
from utils import build_mask


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx=None, trainable=True):
        super(type(self), self).__init__()
        self.n_words = vocab_size
        self.embed_size = embedding_size
        self.embed_layer = nn.Embedding(self.n_words, self.embed_size, padding_idx=padding_idx)
        self.trainable = trainable

    def forward(self, features, normalize=True):
        """
         features is a batch_size * sent_len matrix
        """
        output = self.embed_layer(features)
        if normalize:
            output = self.__normalize_embedding(output)
        if not self.trainable:
            output = Variable(output.data)
        return output

    def __normalize_embedding(self, embed):
        output_norm = torch.norm(embed, 2, 2, keepdim=True)
        embed = embed / output_norm
        return embed

    def get_all_embedding(self):
        return self.embed_layer.weight

    def initialize_embedding(self, init_embed):
        """
         init_embed must be a np.array
        """
        assert type(init_embed) == np.ndarray
        init_embed = torch.Tensor(init_embed)
        self.embed_layer.weight.data = init_embed

    def normalize_all_embedding(self):
        embed = self.embed_layer.weight.data
        embed_norm = torch.norm(embed, 2, 1, keepdim=True)
        embed = embed / embed_norm
        self.embed_layer.weight.data = embed


class BaseRNN(nn.Module):
    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    def __init__(self, vocab_size, max_len, embedding_size, hidden_size, input_dropout_p, dropout_p, n_layers,
                 rnn_cell):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class RNNEncoder(BaseRNN):
    def __init__(self, max_len, embedding_size, hidden_size,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False):
        # EncoderRNN don't need variable vocab_size, so I set it as zero.
        super(RNNEncoder, self).__init__(0, max_len, embedding_size, hidden_size,
                                         input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, input_var, input_lengths=None):
        embedded = self.input_dropout(input_var)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden


def cat_directions(h, bidirectional_encoder):
    """ If the encoder is bidirectional, do the following transformation.
        (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
    """
    if bidirectional_encoder:
        h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h


class TextFeature(nn.Module):
    def __init__(self, opt):
        super(type(self), self).__init__()
        self.opt = opt
        self.pooling_type = self.opt.pooling_type
        self.embedding = Embedding(opt.vocab_size, opt.embedding_size)
        self.encoder = RNNEncoder(opt.max_len, opt.embedding_size, opt.encoder_hidden_size,
                                  input_dropout_p=opt.input_dropout_p, dropout_p=opt.dropout_p,
                                  n_layers=1, bidirectional=opt.bidirectional,
                                  rnn_cell=opt.rnn_type, variable_lengths=opt.variable_encoder_input)
        if self.pooling_type == 3:
            self.lin = nn.Linear(self.opt.classifier_input_size, 300)
            self.lin2 = nn.Linear(300, 1, bias=False)

    def forward(self, seqs_x, lengths):
        seq_embedding = self.embedding(seqs_x, self.opt.normalize_word_embedding)
        output, hidden = self.encoder(seq_embedding, lengths)
        if self.pooling_type == 0:
            output = torch.transpose(output, 1, 2)
            features, _ = torch.max(output, dim=2)
        elif self.pooling_type == 1:
            output = torch.transpose(output, 1, 2)
            features = torch.mean(output, dim=2)
        elif self.pooling_type == 2:
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            hidden = cat_directions(hidden, self.opt.bidirectional)
            hidden = torch.transpose(hidden, 0, 1)
            batch_size = hidden.size(0)
            features = hidden.view(batch_size, -1)
        elif self.pooling_type == 3:
            # Self Attention
            h = F.tanh(self.lin(output))
            h = self.lin2(h)
            h = torch.transpose(h, 0, 1)
            h = F.softmax(h)
            h = torch.transpose(h, 0, 1)
            features = torch.sum(h * output, dim=1)
        else:
            raise NotImplementedError('pooling_type {} is not implemented'.format(self.pooling_type))
        return features


class TextClassifier(nn.Module):
    def __init__(self, opt):
        super(type(self), self).__init__()
        self.opt = opt
        self.hidden = nn.Linear(self.opt.classifier_input_size, self.opt.classifier_hidden_size)
        if self.opt.classifier_use_batch_normalization:
            self.bnorm = nn.BatchNorm1d(self.opt.classifier_hidden_size)
        self.output = nn.Linear(self.opt.classifier_hidden_size, self.opt.classifier_output_size)

    def forward(self, features):
        hidden_features = F.tanh(self.hidden(features))
        if self.opt.classifier_use_batch_normalization:
            hidden_features = self.bnorm(hidden_features)
        result = self.output(hidden_features)
        return result

    def hidden1_L1Norm(self):
        return torch.norm(self.hidden.weight, 1)


if __name__ == '__main__':
    vocab_size = 23
    maxlen = 20
    embedding_size = 30
    batch_size = 2
    hidden_size = 25
    n_layers = 1
    bidirectional = True
    embedding = torch.rand(batch_size, maxlen, embedding_size)
    encoder = RNNEncoder(maxlen, embedding_size, hidden_size, n_layers=n_layers, bidirectional=bidirectional, variable_lengths=True)
    e_output, e_hidden = encoder(embedding, [4,2])
    words = torch.LongTensor([[2,3,4,1],[2,6,1,0]])
    words = Variable(words)
    lenghts = [4, 3]
    from config import Options
    opt = Options()
    opt.vocab_size = 20
    opt.pooling_type = 2
    feature_extractor = TextFeature(opt)
    feature = feature_extractor(words, lenghts)
    print(feature)
