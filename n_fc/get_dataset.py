#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/3 20:08
@File:          get_dataset.py
'''

import random
from pathlib import Path

def get_dataset(root_dir=r'D:\datasets\Tencent-Verification-Code', train_val_split=0.8):
    root_path = Path(root_dir)

    X, Y = [], []
    for sub_path in root_path.iterdir():
        dir = sub_path.as_posix()
        label = sub_path.name[:4]
        X.append(dir)
        Y.append(label)

    total_samples = len(Y)
    index = [i for i in range(total_samples)]
    random.shuffle(index)

    new_X, new_Y = [], []
    for i in index:
        new_X.append(X[i])
        new_Y.append(Y[i])
        
    X, Y = new_X, new_Y

    num_train = int(total_samples * train_val_split)

    return (X[:num_train], Y[:num_train]), (X[num_train:], Y[num_train:])
