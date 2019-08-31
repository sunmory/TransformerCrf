# encoding: utf-8

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../Dataset/train.txt', help='train_dataset')
    parser.add_argument('--dev_path', type=str, default='../Dataset/dev.txt', help='dev_dataset')
    parser.add_argument('--test_path', type=str, default='../Dataset/test.txt', help='test_dataset')
    parser.add_argument('--cha2id_path', type=str, default='../Dataset/ch2id_dict.pkl', help='cha2id_path')
    parser.add_argument('--output_path', type=str, default='../Output/', help='output_path')
    parser.add_argument('--batch_size', type=int, default=50, help='batch_size')
    parser.add_argument('--hidden_size', type=int, default=150, help='hidden_size')
    parser.add_argument('--embedding_size', type=int, default=150, help='embedding_size')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout_rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning_rate')
    parser.add_argument('--max_sequence_length', type=int, default=302, help='batch_size')
    parser.add_argument('--dmodel', type=int, default=256, help='dmodel')
    parser.add_argument('--middle_units', type=int, default=1024, help='middle_units')
    args = parser.parse_args()
    return args