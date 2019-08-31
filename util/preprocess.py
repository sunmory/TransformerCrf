# encoding: utf-8

import pickle as pkl
from model.args import parse_args

args = parse_args()


def generate_dataset():
    with open('../dataset/train_o.txt', 'r', encoding='utf-8') as f:
        dataset = f.readlines()
    # sentences, labels = [], []
    train_dataset = dataset[:int(len(dataset) * 0.9)]
    dev_dataset = dataset[int(len(dataset) * 0.9):]

    def wirte_dataset(dataset_path, dataset):
        with open(dataset_path, 'w', encoding='utf-8') as f:
            for sentence_index in range(len(dataset)):
                sentence, label = [], []
                o_sentence = dataset[sentence_index]
                entity_labels = o_sentence.strip().split()
                for entity_label in entity_labels:
                    entitys, cla = entity_label.split('/')
                    entitys = entitys.split('_')
                    for entity_index in range(len(entitys)):
                        if cla is 'o':
                            sentence.append(entitys[entity_index])
                            label.append('O')
                            continue
                        if entity_index is 0:
                            sentence.append(entitys[entity_index])
                            label.append('B' + cla)
                        elif entity_index is len(entitys) - 1:
                            sentence.append(entitys[entity_index])
                            label.append('E' + cla)
                        else:
                            sentence.append(entitys[entity_index])
                            label.append('I' + cla)
                for e, l in zip(sentence, label):
                    f.write(e + '\t' + l + '\n')
                f.write('\n')
    wirte_dataset(args.train_path, train_dataset)
    wirte_dataset(args.dev_path, dev_dataset)


def generate_ch2id_dict(min_num=3):
    ch_freauency_dict = dict()
    ch_id_dict = dict()
    with open(args.train_path, 'r', encoding='utf-8') as f:
        for l in f:
            w_l = l.strip().split()
            if len(w_l) is 0: continue
            w = w_l[0]
            if w not in ch_freauency_dict.keys():
                ch_freauency_dict[w] = 1
            else:
                ch_freauency_dict[w] = ch_freauency_dict[w] + 1
    ch_id_dict['PAD'] = 0
    ch_id_dict['UNK'] = 1
    id_index = 2
    for w, f in ch_freauency_dict.items():
        if f < min_num: continue
        if w not in ch_id_dict.keys():
            ch_id_dict[w] = id_index
            id_index = id_index + 1
    with open(args.cha2id_path, 'wb') as f:
        pkl.dump(ch_id_dict, f)


if __name__ == '__main__':
    generate_dataset()
    generate_ch2id_dict()