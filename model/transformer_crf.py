# encoding: utf-8

import numpy as np
import tensorflow as tf
from model.crf import Crf
from model.args import parse_args

tf.enable_eager_execution()


class TransformerCrf(tf.keras.Model):
    def __init__(self, args, ch_num, label_num, layer_num):
        super(TransformerCrf, self).__init__()
        self.max_sequence_length = args.max_sequence_length

        self.ch_num = ch_num
        self.embedding_size = args.dmodel
        self.ch_embedding = tf.Variable(initial_value=tf.random_uniform((self.ch_num, self.embedding_size)), trainable=True, dtype=tf.float32, name='char_embedding')
        self.pos_embedding = self.get_position_embedding()
        self.hidden_size = args.hidden_size
        self.learning_rate = args.learning_rate
        self.drop_rate = args.dropout_rate
        self.label_num = label_num
        self.dmodel = args.dmodel

        self.middle_units = args.middle_units
        self.layer_num = layer_num
        self.multiattention_layers = [MultiheadAttention(self.dmodel) for _ in range(self.layer_num)]
        self.gamma = tf.Variable(tf.ones((self.dmodel)), trainable=True, name='gamma', dtype=tf.float32)
        self.biase = tf.Variable(tf.zeros((self.dmodel)), trainable=True, name='gamma', dtype=tf.float32)
        self.feed_forward_dense1 = tf.layers.Dense(self.middle_units)
        self.feed_forward_dense2 = tf.layers.Dense(self.dmodel)
        self.output_dense = tf.layers.Dense(self.label_num)

        self.crf = Crf(self.label_num)

    def get_position_embedding(self):
        position_embedding = np.array([[pos / np.power(10000, (i - i % 2) / self.embedding_size)
                                        for i in range(self.embedding_size)] for pos in range(self.max_sequence_length)])
        position_embedding[:, 0:-1:2] = np.sin(position_embedding[:, 0:-1:2])
        position_embedding[:, 1:-1:2] = np.cos(position_embedding[:, 1:-1:2])
        position_embedding = tf.convert_to_tensor(position_embedding, dtype=tf.float32, name='position_embedding')
        return position_embedding

    def call(self, batch_ch_ids, label, sentence_length, label_list, start_list, end_list, mode='train'):
        batch_ch_encodding = tf.nn.embedding_lookup(self.ch_embedding, batch_ch_ids)
        batch_size, batch_sentence_length = batch_ch_ids.shape[0], batch_ch_ids.shape[1]
        batch_pos_ids = tf.tile(tf.expand_dims(tf.range(batch_sentence_length), 0), [batch_size, 1])
        batch_pos_embedding = tf.nn.embedding_lookup(self.pos_embedding, batch_pos_ids)
        batch_ch_encodding = batch_ch_encodding + batch_pos_embedding
        batch_ch_encodding = tf.layers.dropout(batch_ch_encodding, self.drop_rate)
        for layer_index in range(self.layer_num):
            batch_ch_encodding = self.multi_head_attention(batch_ch_encodding, layer_index, sentence_length)
            batch_ch_encodding = self.feedforward(batch_ch_encodding)
        predict_logits = self.output_dense(batch_ch_encodding)
        if mode is 'train':
            loss = self.crf(predict_logits, label, sentence_length)
            return loss
        if mode is 'dev':
            predict_lables = self.crf.decode(predict_logits, sentence_length, label_list, start_list, end_list)
            return predict_lables

    def layer_normalization(self, batch_input):
        mean, varience = tf.nn.moments(batch_input, axes=-1, keep_dims=True)
        normalized = (batch_input - mean) / (varience + 1e-23)**0.5
        normalized = self.gamma * normalized + self.biase
        return normalized

    def multi_head_attention(self, batch_input, layer_index, sentence_length):
        multiple_embedding = self.multiattention_layers[layer_index](batch_input, sentence_length)
        multiple_embedding = tf.nn.dropout(multiple_embedding, self.drop_rate)
        multiple_embedding += batch_input
        multiple_embedding = self.layer_normalization(multiple_embedding)
        return multiple_embedding

    def feedforward(self, batch_input):
        middle = self.feed_forward_dense1(batch_input)
        output = self.feed_forward_dense2(middle)
        output += batch_input
        output = self.layer_normalization(output)
        return output


class MultiheadAttention(tf.keras.layers.Layer):
    def __init__(self, dmodel):
        super(MultiheadAttention, self).__init__()
        self.dmodel = dmodel
        self.Q_dense = tf.keras.layers.Dense(self.dmodel)
        self.K_dense = tf.keras.layers.Dense(self.dmodel)
        self.V_dense = tf.keras.layers.Dense(self.dmodel)

    def call(self, batch_embedding, sentence_length, head_num=8):
        Q = self.Q_dense(batch_embedding)
        K = self.K_dense(batch_embedding)
        V = self.V_dense(batch_embedding)

        Q = tf.concat(tf.split(Q, head_num, axis=-1), axis=0)
        K = tf.concat(tf.split(K, head_num, axis=-1), axis=0)
        V = tf.concat(tf.split(V, head_num, axis=-1), axis=0)

        scores = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        scores /= self.dmodel ** 0.5
        scores = self.mask(scores, 'K', sentence_length, head_num)
        scores = tf.nn.softmax(scores)
        scores = self.mask(scores, 'Q', sentence_length, head_num)
        # print('scores: {}'.format(scores))
        output = tf.matmul(scores, V)
        output = tf.concat(tf.split(output, head_num, axis=0), axis=-1)
        return output

    def mask(self, input, mode, sentence_length, head_num):
        batch_sentence_num, batch_sequence_length = input.shape[0], input.shape[1]
        mask_bool = tf.sequence_mask(sentence_length)
        # print(mask_bool)
        if mode is 'K':
            mask_bool = tf.expand_dims(mask_bool, 1)
            mask_bool = tf.tile(mask_bool, [head_num, batch_sequence_length, 1])
            ones = tf.ones_like(mask_bool, dtype=tf.float32)
            zeros = ones * (-2 ** 32 + 1)
            input = tf.where(mask_bool, input, zeros)
            assert batch_sentence_num == mask_bool.shape[0]

        if mode is 'Q':
            ones = tf.ones_like(mask_bool, dtype=tf.float32)
            zeros = tf.zeros_like(mask_bool, dtype=tf.float32)
            mask_id = tf.where(mask_bool, ones, zeros)
            mask_id = tf.expand_dims(mask_id, -1)
            mask_id = tf.tile(mask_id, [head_num, 1, batch_sequence_length])
            # print(mask_id)
            input = input * mask_id
            assert batch_sentence_num == mask_id.shape[0]
        return input


if __name__ == '__main__':
    args = parse_args()
    ch_num, label_num, layer_num = 10, 10, 6
    transformer = TransformerCrf(args, ch_num, label_num, layer_num)
    char_ids = np.array([[1, 5, 2, 8, 7, 6, 2, 0, 0, 0], [1, 5, 6, 7, 8, 1, 2, 9, 7, 3]])
    label = np.array([[1, 1, 0, 0, 0, 2, 2, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 2, 2, 2]])
    label_list = ['Ba', 'Ia', 'Ea',  'Bb', 'Ib', 'Eb', 'Bc', 'Ic', 'Ec', 'O']
    start_list = ['Ba', 'Bb', 'Bc', 'O']
    end_list = ['Ea', 'Eb', 'Ec', 'O']

    sentence_length = [7, 10]
    output = transformer(char_ids, label, sentence_length, label_list, start_list, end_list, mode='dev')

    print(output)
