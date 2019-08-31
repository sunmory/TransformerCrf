# encoding: utf-8

import numpy as np

import tensorflow as tf

tf.enable_eager_execution()


class Crf(tf.keras.Model):
    def __init__(self, label_size):
        super(Crf, self).__init__()
        self.label_size = label_size

        self.trans = self.add_variable(name='trans',  shape=[self.label_size, self.label_size], initializer=tf.random_uniform_initializer(), trainable=True)

    def compute_path_score(self, pre_scores, label):
        """
        compute the label path score
        :param pre_scores: [batch_size, sentence_length, label_num] score from LSTM or transformer
        :param label: [batch_size, sentence_length, label_num] one_hot
        :return:
        """
        point_score = tf.reduce_sum(tf.reduce_sum(pre_scores * label, axis=2), 1, keep_dims=True)
        label1 = tf.expand_dims(label[:, :-1], 3)
        label2 = tf.expand_dims(label[:, 1:], 2)
        label = label1 * label2
        trans = tf.expand_dims(tf.expand_dims(self.trans, 0), 0)
        trans_score = tf.reduce_sum(tf.reduce_sum(label * trans, [2, 3]), 1, keep_dims=True)
        return point_score + trans_score

    def comput_z(self, position_input, position_trans_score):
        """
        compute the path score at current time stamp
        :param position_input: [batch_size, label_num] label score on current time stamp
        :param position_trans_score: [batch_size, label_num] path score from last time stamp
        :return:
        """
        position_trans_score = tf.expand_dims(position_trans_score, 2)
        trans = tf.expand_dims(self.trans, 0)
        output = tf.reduce_logsumexp(position_trans_score + trans, 1)
        return output + position_input

    def call(self, input, label, length):
        label = tf.one_hot(label, self.label_size)
        mask = tf.cast(tf.sequence_mask(length), dtype=tf.float32)
        label = label * tf.expand_dims(mask, axis=2)
        input = input * tf.expand_dims(mask, axis=2)
        label_score = self.compute_path_score(input, label)
        batch_sentence_length = input.shape[1]
        pre_trans_score = input[:, 0, :]
        trans_scores_list = [pre_trans_score]
        for i in range(batch_sentence_length - 1):
            pre_trans_score = self.comput_z(input[:, i + 1, :], pre_trans_score)
            trans_scores_list.append(pre_trans_score)
        final_scores = []
        # get full path score depend on sentence length for z computing
        for batch_index, l in enumerate(length):
            final_scores.append(tf.reshape(trans_scores_list[l - 1][batch_index, :], [1, -1]))
        # print(trans_scores_list)
        # print(final_scores)
        final_scores = tf.concat(final_scores, axis=0)
        z = tf.reduce_logsumexp(final_scores, axis=1, keep_dims=True)

        return z - label_score

    # label2ids = {'B': 0, 'I': 1, 'O': 2, 'E': 3, 'S': 5}

    def decode(self, input, length, label_list, start_list, end_list):
        predict_labels = []
        trans_dict = dict()
        for i, l1 in enumerate(label_list):
            for j, l2 in enumerate(label_list):
                trans_dict[l1 + l2] = self.trans[i, j]
        for sentence_index in range(input.shape[0]):
            scores = input[sentence_index, :, :]
            sentence_length = length[sentence_index]
            nodes = [dict(zip(label_list, i)) for i in scores[:sentence_length, :]]
            nodes[0] = {l: s for l, s in nodes[0].items() if l in start_list}
            nodes[-1] = {l: s for l, s in nodes[-1].items() if l in end_list}
            sentence_predict_label = self.viterbi(nodes, trans_dict)
            predict_labels.append(sentence_predict_label)
        return predict_labels

    def viterbi(self, nodes, trans_dict):
        path = nodes[0]
        for l in range(1, len(nodes)):
            path_old, path = path, {}
            for n, ns in nodes[l].items():
                max_score, max_path = -1e10, ''
                for pn, pns in path_old.items():
                    pn = pn.split(' ')
                    current_score = pns + trans_dict[pn[-1] + n] + ns
                    pn.append(n)
                    if current_score > max_score:
                        max_score = current_score
                        max_path = ' '.join(pn)
                path[max_path] = max_score
        max_score, max_path = -1e10, ''
        for l, s in path.items():
            if s > max_score:
                max_score = s
                max_path = l

        return max_path.split(' ')


if __name__ == '__main__':

    label_size = 3
    batch_size = 2

    sentence_length = [5, 6]
    batch_sentence_length = 6
    # input = tf.Variable(initial_value=tf.random_uniform(shape=[batch_size, label_size]), dtype=tf.float32)
    input = np.random.normal(0, 1, (batch_size, batch_sentence_length, label_size)).astype(np.float32)
    label = tf.convert_to_tensor([[0, 1, 1, 2, 0, 0], [0, 1, 0, 2, 0, 1]])

    # label1 = [1, 2]
    length = tf.convert_to_tensor(sentence_length)

    crf = Crf(label_size)

    loss_fn = tf.keras.losses.sparse_categorical_crossentropy

    # print(r)
    # optimizer = tf.train.AdamOptimizer(0.001)
    optimizer = tf.keras.optimizers.SGD(0.001)

    with tf.GradientTape() as tape:
        # loss = tf.reduce_mean(loss)
        loss = crf(input, label, length)
        # loss = loss_fn(label1, r)
        loss = tf.reduce_mean(loss)
        print(loss)
    grads = tape.gradient(loss, crf.trainable_variables)
    optimizer.apply_gradients(zip(grads, crf.trainable_variables))




