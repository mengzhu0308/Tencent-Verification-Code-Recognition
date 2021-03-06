#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/15 20:27
@File:          ToOneHot.py
'''

import numpy as np

class ToOneHot:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, label):
        out = []
        for l in label:
            one_hot = np.zeros(self.num_classes, dtype='int32')
            one_hot[l] = 1
            out.append(one_hot)
        return np.concatenate(out)