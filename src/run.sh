#!/usr/bin/env bash
tar xzf data_sample.tar.gz
python build_all_dataset.py --primary_datapath data_sample/
python preprocess_data_cnn.py
python train_cnn_classifier_F1.py --feature_net_path emb_cnn.pt --classifier_net_path cnn.pt --pooling_type 0 --classifier_lr 0.001 --batch_size 128 --normalize_word_embedding 0
