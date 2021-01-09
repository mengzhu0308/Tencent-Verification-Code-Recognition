#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/16 6:52
@File:          generator.py
'''

import random
import numpy as np

def generator(dataset, batch_size=64, shuffle=True, drop_last=False):
    image, label = dataset[0]
    image_size = image.shape

    true_examples = len(dataset)
    rd_index = [i for i in range(true_examples)]

    false_examples = true_examples // batch_size * batch_size
    remain_examples = true_examples - false_examples

    i = 0
    while True:
        real_batch_size = batch_size
        if remain_examples != 0 and drop_last is False and i == false_examples:
            real_batch_size = remain_examples

        batch_label_lengths = np.ones(real_batch_size, dtype='int32') * dataset.label_length
        batch_input_lengths = np.ones(real_batch_size, dtype='int32') * dataset.input_length
        batch_images = np.empty((real_batch_size, *image_size), dtype='float32')
        batch_labels = np.empty((real_batch_size, dataset.label_length), dtype='int32')

        for b in range(real_batch_size):
            if shuffle and i == 0:
                random.shuffle(rd_index)
                image_dirs, labels = [], []
                for i in rd_index:
                    image_dirs.append(dataset.image_dirs[i])
                    labels.append(dataset.labels[i])
                dataset.image_dirs = image_dirs
                dataset.labels = labels

            batch_images[b], batch_labels[b] = dataset[i]

            if remain_examples != 0 and drop_last is True:
                i = (i + 1) % false_examples
            else:
                i = (i + 1) % true_examples

        yield [batch_label_lengths, batch_input_lengths, batch_labels, batch_images], None