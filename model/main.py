# encoding: utf-8

import os
import datetime
import pickle as pkl
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from model.transformer_crf import TransformerCrf
from model.args import parse_args

args = parse_args()


with open(args.cha2id_path, 'rb') as f:
    char2ids = pkl.load(f)

label2ids = {'Ba': 0, 'Ia': 1, 'Ea': 2,  'Bb': 3, 'Ib': 4, 'Eb': 5, 'Bc': 6, 'Ic': 7, 'Ec': 8, 'O': 9}
# label2ids2 = {'B': 0, 'I': 1, 'E': 2, 'O': 3}


def padding(dataset, label):
    max_sentence_length = 0
    dataset_length, kb_matched_num = [], []
    for sentence in dataset:
        if len(sentence) > max_sentence_length:
            max_sentence_length = len(sentence)

    for sentence_index in range(len(dataset)):
        dataset_length.append(len(dataset[sentence_index]))
        dataset[sentence_index].extend(['PAD'] * (max_sentence_length - len(dataset[sentence_index])))
        label[sentence_index].extend(['O'] * (max_sentence_length - len(label[sentence_index])))
    return dataset, label, dataset_length


def change2id(dataset, label):
    for sentence_index in range(len(dataset)):
        sentence_ch_ids = [char2ids.get(ch, 1) for ch in dataset[sentence_index]]
        sentence_label_ids = [label2ids[l] for l in label[sentence_index]]
        dataset[sentence_index] = sentence_ch_ids
        label[sentence_index] = sentence_label_ids

    return np.array(dataset), np.array(label)


def read_dataset(dataset_path):
    sentences, labels = [], []
    sentence, label = [], []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for l in f:
            w_l = l.strip().split()
            if len(w_l) > 0:
                sentence.append(w_l[0])
                label.append(w_l[1])
            else:
                if len(sentence) <= max_sentence_length:
                    sentences.append(sentence)
                    labels.append(label)
                sentence, label = [], []
    return sentences, labels


def batch_generate(mode='train'):
    if mode is 'train':
        data_path = args.train_path
    if mode is 'dev':
        data_path = args.dev_path
    batch_size = args.batch_size
    sentences, labels = read_dataset(data_path)
    for index in tqdm(list(range(0, len(sentences), batch_size))):
        batch_sentecnes, batch_labels = sentences[index: index + batch_size], labels[index: index + batch_size]
        # print(batch_sentecnes)
        # print(char2ids)
        batch_sentecnes, batch_labels, batch_length = padding(batch_sentecnes, batch_labels)
        batch_sentecnes, batch_labels = change2id(batch_sentecnes, batch_labels)
        yield(batch_sentecnes, batch_labels, batch_length)


def train(epoch_num=1):
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    model_path = os.path.join(args.output_path, time_str + "/")
    optimizer = tf.train.AdamOptimizer(args.learning_rate)

    batch_index, best_F1, max_length = 0, 0, 0
    for epoch in range(epoch_num):
        batch_generator = batch_generate(mode='train')
        try:
            while True:

                batch_sentence_ids, batch_label_ids, batch_length = next(batch_generator)
                # l = batch_sentence_ids.shape[1]
                # if l > max_length:
                #     max_length = l
                print('batch_index: {}, batch_length: {}'.format(batch_index, batch_sentence_ids.shape))
                with tf.GradientTape() as tape:
                    loss = transofrmer_crf(batch_sentence_ids, batch_label_ids, batch_length, label_list, start_list, end_list, mode='train')
                    loss = tf.reduce_mean(loss)
                grads = tape.gradient(loss, transofrmer_crf.trainable_variables)
                optimizer.apply_gradients(zip(grads, transofrmer_crf.trainable_variables))
                print("epoch: {} batch: {} loss: {}".format(epoch, batch_index, loss))
                # if not os.path.exists(model_path):
                #     os.makedirs(model_path)
                # transofrmer_crf.save_weights(model_path)
                if (batch_index + 1) % 500 is 0:
                    P, R, F1 = evaluate()
                    print('epoch: {}, batch_index: {}, P: {}, R: {}, F1: {}, best_F1: {}'.format(epoch, batch_index, P, R, F1, best_F1))
                    if F1 > best_F1:
                        print('save model')
                        if not os.path.exists(model_path):
                            os.makedirs(model_path)
                        transofrmer_crf.save_weights(model_path)
                        best_F1 = F1
                batch_index = batch_index + 1
                # if batch_index > 163: break
        except StopIteration as e:
            print('finish epoch: {}'.format(epoch))


def compute_prc_num(predict_label, real_label, mode='l1'):
    if mode is 'l1':
        match_dict = {0: [[0, 1, 2], 'a'], 3: [[3, 4, 5], 'b'], 6: [[6, 7, 8], 'c']}
    else:
        match_dict = {0: [0, 1, 2]}
    p_n, r_n, c_n = 0, 0, 0

    def find_entity(label):
        entity_set = set()
        entity, match_list = list(),  None
        for ch_index in range(len(label)):
            ch = label[ch_index]
            if match_list is None:
                if ch in match_dict.keys():
                    match_list = match_dict[ch][0]
                    entity.append(match_dict[ch][1])
                    entity.append(ch_index)
            if match_list is not None:
                if ch not in match_list:
                    entity.append(ch_index - 1)
                    entity_set.add(tuple(entity))
                    entity, match_list = list(), None
                    if ch in match_dict.keys():
                        match_list = match_dict[ch][0]
                        entity.append(match_dict[ch][1])
                        entity.append(ch_index)
        return entity_set

    for p_l, r_l in zip(predict_label, real_label):
        p_s = find_entity(p_l)
        r_s = find_entity(r_l)
        p_n = p_n + len(p_s)
        r_n = r_n + len(r_s)
        c_n = c_n + len(p_s.intersection(r_s))

    return p_n, r_n, c_n


def evaluate():
    batch_generator = batch_generate(mode='dev')
    total_p_n, total_r_n, total_c_n = 0, 0, 0
    try:

        while True:
            batch_sentence_ids, batch_label_ids, batch_length = next(batch_generator)
            predict_labels = transofrmer_crf(batch_sentence_ids, batch_label_ids, batch_length, label_list, start_list, end_list, mode='dev')
            predict_label_ids, real_label_ids = [], []
            for p_l, r_l, sequence_length in zip(predict_labels, batch_label_ids.tolist(), batch_length):
                p_l = [label2ids[l] for l in p_l[:sequence_length]]
                r_l = r_l[:sequence_length]
                predict_label_ids.append(p_l)
                real_label_ids.append(r_l)
                if total_r_n is 0:
                    print(p_l)
                    print(r_l)

            p_n, r_n, c_n = compute_prc_num(predict_label_ids, real_label_ids)
            total_p_n = total_p_n + p_n
            total_r_n = total_r_n + r_n
            total_c_n = total_c_n + c_n
    except StopIteration as e:

        print('finish evaluate')
    P = total_c_n / (total_p_n + 1)

    R = total_c_n / (total_r_n + 1)
    F1 = 2 * P * R / (P + R + 1e-10)
    return P, R, F1


if __name__ == '__main__':
    max_sentence_length = 300
    label_list = ['Ba', 'Ia', 'Ea',  'Bb', 'Ib', 'Eb', 'Bc', 'Ic', 'Ec', 'O']
    start_list = ['Ba', 'Bb', 'Bc', 'O']
    end_list = ['Ea', 'Eb', 'Ec', 'O']
    transofrmer_crf = TransformerCrf(args, len(char2ids.keys()), len(label2ids.keys()), 3)
    print('begin')
    train(1500)