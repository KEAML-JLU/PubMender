"""
 author:hzhanght
"""
from __future__ import print_function, absolute_import, division
import os
import random
import re
import io
import pickle
import config

opt = config.Options(config_vocab=False, config_class=False)
primary_data_path = opt.primary_datapath
dataset_path = opt.data_path
class_config_path = opt.class_config_path
idx2label = {i: n for i, n in enumerate(os.listdir(primary_data_path))}
label2idx = {n: i for i, n in idx2label.items()}
with open(class_config_path, 'wb') as f:
    pickle.dump((label2idx, idx2label), f)


def build_dataset_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
train_dataset_path = os.path.join(dataset_path, 'train')
valid_dataset_path = os.path.join(dataset_path, 'valid')
test_dataset_path = os.path.join(dataset_path, 'test')
build_dataset_dir(train_dataset_path)
build_dataset_dir(valid_dataset_path)
build_dataset_dir(test_dataset_path)


def get_abstract(text):
    def tmp(line):
        lst = ['bold:', 'italic:', 'sup:']
        result = True
        for l in lst:
            if line.startswith(l):
                result = False
                break
        return result
    lines = text.splitlines()
    result = []
    flag = False
    for line in lines:
        line = line.strip()
        if line.startswith('abstract:'):
            line = line[9:]
            result.append(line)
            flag = True
        elif line.startswith('pub_date:'):
            break
        elif 'title:' in line and len(line) < 20:
            continue
        elif flag and tmp(line):
            result.append(line)
    return ' '.join(result)


def process_text(text):
    text = text.strip()
    text = re.sub(r"[\[\]{}:_`]", " ", text)
    return text


for journal_name in os.listdir(primary_data_path):
    journal_path = os.path.join(primary_data_path, journal_name)
    paper_lists = os.listdir(journal_path)
    random.shuffle(paper_lists)

    all_data_size = len(paper_lists)
    train_num = int(0.8 * all_data_size)
    val_num = int(0.1 * all_data_size)
    test_num = all_data_size - train_num - val_num
    print('{} train size {} val size {} test size {}'.format(journal_name, train_num, val_num, test_num))

    train_lists = paper_lists[:train_num]
    test_lists = paper_lists[train_num:train_num+test_num]
    valid_lists = paper_lists[train_num+test_num:train_num+test_num+val_num]
    for part in ['train', 'test', 'valid']:
        tmp_path = os.path.join(eval(part+'_dataset_path'), journal_name)
        tmp_lists = eval(part+'_lists')
        with io.open(tmp_path, 'w', encoding='utf-8') as f:
            for paper_name in tmp_lists:
                if not paper_name.endswith('.txt'):
                    break
                p = os.path.join(journal_path, paper_name)
                with io.open(p, 'r', encoding='utf-8') as fr:
                    text = fr.read()
                #text = get_abstract(text)
                text = text.strip()
                text = process_text(text)
                if len(text) < 150:
                    continue
                f.write(text + '\n')
