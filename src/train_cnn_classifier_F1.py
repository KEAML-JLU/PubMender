"""
 author: hzhanght
"""
from __future__ import print_function, absolute_import, division
import torch
from torch.autograd import Variable
import numpy as np
import os
import cPickle as pickle
from models import Embedding
from config import Options
from utils import get_minibatches_idx, rec_sents, batch_to_sents, split_data_by_label
import ML_CNN
from Fscore import FScore

topK = (1, 3, 5, 10, 20)
important_K = 0
assert 0 <= important_K < len(topK)

def prepare_data_for_cnn(seqs_x, labels=None, sent_len=350):
    n_samples = len(seqs_x)
    # 0 is Padding token idx
    seqs_x = [x[:sent_len] for x in seqs_x]
    x = np.zeros(( n_samples, sent_len)).astype('int32')
    for idx, s_x in enumerate(seqs_x):
        x[idx, :len(s_x)] = s_x
    x = torch.from_numpy(x).long()
    if labels is not None:
        labels = torch.LongTensor(labels)
    return x, labels


def test_result(data, labels, topK=(1, 3, 5, 10, 20)):
    labels = labels.view(-1,1)
    _, predict = torch.sort(data, dim=1, descending=True)
    result = []
    for k in topK:
        tmp_predict = predict[:, :k]
        tmp_result = (labels == tmp_predict).long()
        tmp_result = torch.sum(tmp_result)
        result.append(tmp_result)
    return result


def update_F1(data, labels, label_size, topK, result_tables):
    labels_one = labels.view(-1,1)
    _, predict = torch.sort(data, dim=1, descending=True)
    result = []
    for k in topK:
        tmp_predict = predict[:, :k]
        tmp_result = (labels_one == tmp_predict).long()
        tmp_result = torch.sum(tmp_result, dim=1)
        for label_id in range(label_size):

            tp = torch.sum((labels == label_id).long() * tmp_result)
            fn = torch.sum((labels == label_id).long() * (1 - tmp_result))
            fp = torch.sum(torch.sum(tmp_predict == label_id, dim=1).long() * (1 - (labels == label_id).long()) * (1 - tmp_result))
            fscore_obj = result_tables[label_id][k]
            fscore_obj.add_tp(tp=tp)
            fscore_obj.add_fp(fp=fp)
            fscore_obj.add_fn(fn=fn)


def main():
    opt = Options()
    print('Use {}'.format(opt.pooling_type_str_dict[opt.pooling_type]))
    train_sents, train_labels = pickle.load(open(opt.train_path, 'rb'))
    valid_sents, valid_labels = pickle.load(open(opt.valid_path, 'rb'))
    test_sents, test_labels = pickle.load(open(opt.test_path, 'rb'))
    #
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    #

    emb = Embedding(opt.vocab_size, 200, padding_idx=0, trainable=False)
    cnn = ML_CNN.CNN_Module(n_classes=opt.classifier_output_size)

    if opt.use_cuda:
        emb.cuda()
        cnn.cuda()
    param = []
    param.extend(emb.parameters())
    param.extend(cnn.parameters())
    # optimizer = torch.optim.Adam(param, lr=opt.lr, weight_decay=0.01)
    # optimizer = torch.optim.Adam(param, lr=opt.lr, weight_decay=0.00001)
    optimizer = torch.optim.Adam(param, lr=opt.lr)
    criteron = torch.nn.CrossEntropyLoss()

    if opt.restore:
        if os.path.exists(opt.feature_net_path):
            print("Load pretrained embedding")
            emb.load_state_dict(torch.load(opt.feature_net_path))
        else:
            print("No pretrained embedding")
        if os.path.exists(opt.classifier_net_path):
            print("Load pretrained cnn classifier")
            cnn.load_state_dict(torch.load(opt.classifier_net_path))
        else:
            print("No pretrained cnn classifier")

    best_acc = -1
    for epoch in range(opt.max_epochs):
        print("Starting epoch %d" % epoch)
        kf = get_minibatches_idx(len(train_sents), opt.batch_size, shuffle=True)
        epoch_losses = []
        cnn.train()
        emb.train()
        for iteridx, train_index in kf:
            if len(train_index) <= 1:
                continue
            sents = [train_sents[t] for t in train_index]
            labels = [train_labels[t] for t in train_index]
            # X_batch, X_lengths, X_labels = prepare_data_for_rnn(sents, labels)
            X_batch, X_labels = prepare_data_for_cnn(sents, labels)
            X_batch = Variable(X_batch)
            X_labels = Variable(X_labels)
            if opt.use_cuda:
                X_batch = X_batch.cuda()
                X_labels = X_labels.cuda()
            optimizer.zero_grad()
            features = emb(X_batch)
            output = cnn(features)
            loss = criteron(output, X_labels)
            local_loss = loss.data[0]
            epoch_losses.append(local_loss)
            loss.backward()
            optimizer.step()
            if iteridx % opt.print_freq == 0:
                count = output.size(0)
                topK_correct = test_result(output.cpu().data, X_labels.cpu().data, topK=topK)
                topK_acc = [float(tmp) / count for tmp in topK_correct]
                topK_str = " , ".join(["acc@{}: {}".format(k, tmp_acc) for k, tmp_acc in zip(topK, topK_acc)])
                print("Epoch {} Iteration {}  loss: {} , {}".format(epoch + 1, iteridx + 1, local_loss, topK_str))

        ave_loss = sum(epoch_losses) / len(epoch_losses)
        kf = get_minibatches_idx(len(valid_sents), opt.batch_size, shuffle=True)
        count = 0
        all_topK_correct = np.zeros(len(topK), dtype=int)
        for _, valid_index in kf:
            emb.eval()
            cnn.eval()
            sents = [valid_sents[t] for t in valid_index]
            labels = [valid_labels[t] for t in valid_index]
            X_batch, X_labels = prepare_data_for_cnn(sents, labels)
            X_batch = Variable(X_batch)
            X_labels = Variable(X_labels)
            if opt.use_cuda:
                X_batch = X_batch.cuda()
                X_labels = X_labels.cuda()
            features = emb(X_batch)
            output = cnn(features)
            topK_correct = test_result(output.cpu().data, X_labels.cpu().data, topK=topK)
            topK_correct = np.array(topK_correct)
            all_topK_correct += topK_correct
            bsize = output.size(0)
            count += bsize

        all_topK_acc = all_topK_correct / float(count)
        all_topK_acc = all_topK_acc.tolist()
        all_topK_str = " , ".join(["val_acc@{}: {}".format(k, tmp_acc) for k, tmp_acc in zip(topK, all_topK_acc)])
        print("Epoch {} Avg_loss: {}, {}".format(epoch+1, ave_loss, all_topK_str))
        acc = all_topK_acc[important_K]
        if acc > best_acc:
            print('Dump current model due to current acc {} > past best acc {}'.format(acc, best_acc))
            torch.save(cnn.state_dict(), opt.classifier_net_path)
            best_acc = acc

        fscore_records = [{k:FScore() for k in topK} for i in range(opt.classifier_output_size)]
        kf = get_minibatches_idx(len(test_sents), opt.batch_size, shuffle=True)
        emb.eval()
        cnn.eval()
        for _, test_index in kf:
            sents = [test_sents[t] for t in test_index]
            labels = [test_labels[t] for t in test_index]
            X_batch, X_labels = prepare_data_for_cnn(sents, labels)
            X_batch = Variable(X_batch)
            X_labels = Variable(X_labels)
            if opt.use_cuda:
                X_batch = X_batch.cuda()
                X_labels = X_labels.cuda()
            features = emb(X_batch)
            output = cnn(features)
            update_F1(output.cpu().data, X_labels.cpu().data, opt.classifier_output_size, topK, fscore_records)
        with open('F_score_dir/{}.pkl'.format(epoch+1),'w') as f:
            print('dumping fscore in epoch {}'.format(epoch+1))
            pickle.dump(fscore_records, f)


    print('Loading best model')
    cnn.load_state_dict(torch.load(opt.classifier_net_path))
    print('Testing Data')
    kf = get_minibatches_idx(len(test_sents), opt.batch_size, shuffle=True)
    count = 0
    all_topK_correct = np.zeros(len(topK), dtype=int)
    fscore_records = [{k:FScore() for k in topK} for i in range(opt.classifier_output_size)]
    for _, test_index in kf:
        emb.eval()
        cnn.eval()
        sents = [test_sents[t] for t in test_index]
        labels = [test_labels[t] for t in test_index]
        X_batch, X_labels = prepare_data_for_cnn(sents, labels)
        X_batch = Variable(X_batch)
        X_labels = Variable(X_labels)
        if opt.use_cuda:
            X_batch = X_batch.cuda()
            X_labels = X_labels.cuda()
        features = emb(X_batch)
        output = cnn(features)
        update_F1(output.cpu().data, X_labels.cpu().data, opt.classifier_output_size, topK, fscore_records)
        topK_correct = test_result(output.cpu().data, X_labels.cpu().data, topK=topK)
        topK_correct = np.array(topK_correct)
        all_topK_correct += topK_correct
        bsize = output.size(0)
        count += bsize
    all_topK_acc = all_topK_correct / float(count)
    all_topK_acc = all_topK_acc.tolist()
    all_topK_str = " , ".join(["test_acc@{}: {}".format(k, tmp_acc) for k, tmp_acc in zip(topK, all_topK_acc)])
    print("Training end {}".format(all_topK_str))

    with open('F_score_dir/best.pkl','w') as f:
        print('dumping fscore in')
        pickle.dump(fscore_records, f)

if __name__ == '__main__':
    main()
