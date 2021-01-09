#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/8 19:02
@File:          train_ctc.py
'''

import math
import numpy as np
from keras import backend as K
from keras.layers import Input
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import Callback

from Loss import Loss
from Dataset import Dataset
from generator import generator
from get_dataset import get_dataset
from ImageTf import ImageTf
from ctc_model import CTC_Model

num_classes = 26
seq_len = 4

class CTCLoss(Loss):
    def compute_loss(self, inputs):
        label_length, input_length, y_true, y_pred = inputs

        loss = K.ctc_batch_cost(y_true, K.softmax(y_pred), input_length, label_length)

        return K.mean(loss)

if __name__ == '__main__':
    train_batch_size = 128
    val_batch_size = 500
    image_size = (53, 129, 3)

    (X_train, Y_train), (X_val, Y_val) = get_dataset()

    train_dataset = Dataset(X_train, Y_train, image_transform=ImageTf(image_size[:2]))
    val_dataset = Dataset(X_val, Y_val, image_transform=ImageTf(image_size[:2]))
    train_generator = generator(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_generator = generator(val_dataset, batch_size=val_batch_size, shuffle=False)

    image_input = Input(shape=image_size, name='image_input', dtype='float32')
    label_length = Input(shape=(1, ), dtype='int32')
    input_length = Input(shape=(1, ), dtype='int32')
    y_true = Input(shape=(seq_len, ), dtype='int32')
    out = CTC_Model(image_input, num_classes=num_classes)
    out = CTCLoss(output_axis=-1)([label_length, input_length, y_true, out])
    model = Model([label_length, input_length, y_true, image_input], out)
    model.compile(Adam())

    num_val_examples = len(Y_val)
    num_val_batches = math.ceil(num_val_examples / val_batch_size)

    def _remove_repeats(inds):
        is_not_repeat = np.insert(np.diff(inds).astype(np.bool), 0, True)
        return inds[is_not_repeat]

    def _remove_blanks(inds, num_classes):
        return inds[inds < (num_classes - 1)]

    def evaluate(model):
        total_loss = 0.
        total_corrects = 0

        for _ in range(num_val_batches):
            batch_data, _ = next(val_generator)

            val_loss, predict = model.test_on_batch(batch_data, y=None), model.predict_on_batch(batch_data)

            for (y_true, y_pred, input_length) in zip(batch_data[2], predict, batch_data[1]):
                decode = np.argmax(y_pred[:input_length], axis=-1)
                decode = _remove_repeats(decode)
                decode = _remove_blanks(decode, num_classes)

                if decode.size == y_true.size and np.sum(decode == y_true) == y_true.size:
                    total_corrects += 1

            total_loss += val_loss

        val_loss = total_loss / num_val_batches
        val_acc = (total_corrects / num_val_examples) * 100

        return val_loss, val_acc

    class Evaluator(Callback):
        def __init__(self):
            super(Evaluator, self).__init__()

        def on_epoch_end(self, epoch, logs=None):
            val_loss, val_acc = evaluate(self.model)

            print(f'val_loss = {val_loss:.5f}, val_acc = {val_acc:.2f}.')

    evaluator = Evaluator()

    model.fit_generator(
        train_generator,
        steps_per_epoch=math.ceil(len(Y_train) / train_batch_size),
        epochs=10,
        callbacks=[evaluator],
        shuffle=False,
        initial_epoch=0
    )