#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2020/12/16 6:51
@File:          Dataset.py
'''

from skimage import io
import numpy as np

def str2ID(s):
    id = [ord(c) - ord('a') for c in s]
    return id

class Dataset:
    def __init__(self, image_dirs, labels, label_len=4, input_len=9, image_transform=None, label_transform=None):
        self.__label_len = label_len
        self.__input_len = input_len
        self.image_dirs = image_dirs
        self.labels = labels
        self.__image_transform = image_transform
        self.__label_transform = label_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = io.imread(self.image_dirs[item])
        label = str2ID(self.labels[item])
        if self.__image_transform is not None:
            image = self.__image_transform(image)
        if self.__label_transform is not None:
            label = self.__label_transform(label)

        return image, np.array(label, dtype='int32')

    @property
    def label_length(self):
        return self.__label_len

    @property
    def input_length(self):
        return self.__input_len