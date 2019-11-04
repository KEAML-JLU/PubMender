"""
 author: hzhanght
"""
import torch
import argparse
import cPickle as pickle

parser = argparse.ArgumentParser(description='Deep Text Cluster Project')

parser.add_argument("--primary_datapath", type=str, default='../file_more_than_2000/', help="primary data path")
parser.add_argument("--datapath", type=str, default='./pubmed', help="Text classification data path")
parser.add_argument("--class_config_path", type=str, default='./data/class_config.p',
                    help="Maps between class and label num")
parser.add_argument("--trainpath", type=str, default='data/train.p', help="Preprocessed train data path")
parser.add_argument("--validpath", type=str, default='data/valid.p', help="Preprocessed valid data path")
parser.add_argument("--testpath", type=str, default='data/test.p', help="Preprocessed test data path")
parser.add_argument("--vocabpath", type=str, default='data/vocab.p', help="Vocabulary path")
parser.add_argument("--min_word_count", type=int, default=10, help="minimum frequency of word")

parser.add_argument("--feature_net_path", type=str, default='feature_extractor.pt', help='Trained feature_extractor network path')
parser.add_argument("--classifier_net_path", type=str, default='classifier.pt', help='Trained classifier network path')

parser.add_argument("--bidirectional", type=int, default=1, help="Bidirectional RNN (>0 True, <=0 False)")
parser.add_argument("--embedding_size", type=int, default=300, help="Dim of word embedding")
parser.add_argument("--rnn_type", type=str, default='lstm', help="Type of RNN cell (gru or lstm)")
parser.add_argument("--encoder_hidden_size", type=int, default=300, help="Dim of rnn hidden layer")
parser.add_argument("--classifier_hidden_size", type=int, default=200, help="Dim of classifier hidden layer")
parser.add_argument("--classifier_batch_norm", type=int, default=1, help="Use batch normalization (>0 True, <=0 False)")
parser.add_argument("--pooling_type", type=int, default=0,
                    help="pooling type (0 max-pooling 1 mean-pooling 2 last hidden layer)")

parser.add_argument("--variable_encoder_input", type=int, default=1,
                    help='variable input for encoder (>0 True, <=0 False)')
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3, help="Default learning rate")
parser.add_argument("--classifier_lr", type=float, default=1e-4, help="Default learning rate")
parser.add_argument("--dropout_p", type=float, default=0.5, help="Encoder and Decoder output dropout")
parser.add_argument("--input_dropout_p", type=float, default=0.1, help="Embedding dropout")
parser.add_argument("--max_norm", type=float, default=5., help="Max norm (grad clipping)")
parser.add_argument("--normalize_word_embedding", type=int, default=1,
                    help="Normlize word embedding (>0 True, <=0 False)")
parser.add_argument("--print_freq", type=int, default=100, help="Frequency of printing train information")
parser.add_argument("--use_cuda", type=int, default=1, help="Whether use gpu (>0 True, <=0 False)")
parser.add_argument("--max_len", type=int, default=600, help="Maximum length of input sentence")
parser.add_argument("--restore", type=int, default=1, help="Restore pretrained model (>0 True, <=0 False)")
parser.add_argument("--substitution", type=str, default='m', help="Add noise into sentence (s, p, a, d, m or `other`)")
parser.add_argument("--permutation", type=int, default=0, help="Changes count of sentence's words")

args = parser.parse_args()


class Options(object):
    def __init__(self, config_vocab=True, config_class=True):
        self.primary_datapath = args.primary_datapath
        self.data_path = args.datapath
        self.class_config_path = args.class_config_path
        self.train_path = args.trainpath
        self.valid_path = args.validpath
        self.test_path = args.testpath
        self.vocab_path = args.vocabpath
        self.feature_net_path = args.feature_net_path
        self.classifier_net_path = args.classifier_net_path
        self.min_word_count = args.min_word_count
        self.vocab = None

        if config_vocab:
            self.load_vocab()
            self.vocab_size = len(self.vocab)

        self.pooling_type = args.pooling_type
        assert 0 <= self.pooling_type <= 3
        self.pooling_type_str_dict = {0: 'max-pooling',
                                 1: 'mean-pooling',
                                 2: 'last hidden',
                                 3: 'self-attention'}
        self.bidirectional = args.bidirectional
        self.embedding_size = args.embedding_size
        self.encoder_hidden_size = args.encoder_hidden_size
        self.classifier_input_size = self.encoder_hidden_size * 2 if self.bidirectional else self.encoder_hidden_size
        self.classifier_hidden_size = args.classifier_hidden_size
        self.classifier_output_size = None
        self.label2idx = None
        self.idx2label = None
        if config_class:
            self.load_class()
            self.classifier_output_size = len(self.label2idx)

        self.classifier_use_batch_normalization = args.classifier_batch_norm > 0
        self.rnn_type = args.rnn_type
        assert self.rnn_type in ['gru', 'lstm'], "rnn_type must be gru or lstm"

        self.variable_encoder_input = args.variable_encoder_input > 0
        self.max_epochs = args.max_epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.dropout_p = args.dropout_p
        self.input_dropout_p = args.input_dropout_p
        self.max_norm = args.max_norm
        self.normalize_word_embedding = args.normalize_word_embedding > 0
        self.print_freq = args.print_freq
        self.use_cuda = torch.cuda.is_available() and args.use_cuda > 0
        self.max_len = args.max_len
        self.restore = args.restore > 0
        self.permutation = args.permutation
        self.substitution = args.substitution
        # Add noise into sentence if self.permutation > 0
        self.use_noise = self.permutation > 0

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value

    def load_vocab(self):
        with open(self.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

    def load_class(self):
        with open(self.class_config_path, 'rb') as f:
            x = pickle.load(f)
            self.label2idx = x[0]
            self.idx2label = x[1]
            assert len(self.label2idx) == len(self.idx2label)


if __name__ == '__main__':
    opt = Options()
    print dict(opt)
