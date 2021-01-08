'''
@Author:        zm
@Date and Time: 2019/8/8 15:14
@File:          train.py
'''

import math
import numpy as np
import tensorflow as tf
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
from ToOneHot import ToOneHot
from ocr_model import OCR_Model

seq_len = 4

class CrossEntropy(Loss):
    def compute_loss(self, inputs, seq_len=seq_len):
        y_true, y_pred = inputs
        y_trues, y_preds = tf.split(y_true, seq_len, axis=-1), tf.split(y_pred, seq_len, axis=-1)

        total_loss = 0.
        for (y_true, y_pred) in zip(y_trues, y_preds):
            loss = K.categorical_crossentropy(y_true, K.softmax(y_pred, axis=-1))
            loss = K.mean(loss)
            total_loss += loss

        return total_loss / seq_len

if __name__ == '__main__':
    train_batch_size = 16
    val_batch_size = 100
    num_classes = 26
    image_size = (53, 129, 3)

    (X_train, Y_train), (X_val, Y_val) = get_dataset()

    train_dataset = Dataset(X_train, Y_train, image_transform=ImageTf(image_size[:2]),
                            label_transform=ToOneHot(num_classes))
    val_dataset = Dataset(X_val, Y_val, image_transform=ImageTf(image_size[:2]), label_transform=ToOneHot(num_classes))
    train_generator = generator(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_generator = generator(val_dataset, batch_size=val_batch_size, shuffle=False)

    image_input = Input(shape=image_size, name='image_input', dtype='float32')
    y_true = Input(shape=(num_classes * seq_len, ), dtype='int32')
    out = OCR_Model(image_input, num_classes, seq_len=seq_len)
    out = CrossEntropy(-1)([y_true, out])
    model = Model([y_true, image_input], out)
    model.compile(Adam())

    num_val_examples = len(Y_val)
    num_val_batches = math.ceil(num_val_examples / val_batch_size)

    def evaluate(model, seq_len=seq_len):
        total_loss = 0.
        total_corrects = 0

        for _ in range(num_val_batches):
            batch_data, _ = next(val_generator)
            val_loss, predict = model.test_on_batch(batch_data, y=None), model.predict_on_batch(batch_data)

            total_loss += val_loss
            y_trues = np.split(batch_data[0], seq_len, axis=-1)
            y_preds = np.split(predict, seq_len, axis=-1)
            tmp = 1
            for (y_true, y_pred) in zip(y_trues, y_preds):
                tmp *= (np.argmax(y_true, axis=-1) == np.argmax(y_pred, axis=-1))
            total_corrects += np.sum(tmp)

        val_loss = total_loss / num_val_batches
        val_acc = (total_corrects / num_val_examples) * 100

        return val_loss, val_acc

    class Evaluator(Callback):
        def __init__(self):
            super(Evaluator, self).__init__()

        def on_epoch_end(self, epoch, logs=None):
            val_loss, val_acc = evaluate(self.model)
            print(f'val_loss = {val_loss:.5f}, top-1 val_acc = {val_acc:.2f}.')

    evaluator = Evaluator()

    model.fit_generator(
        train_generator,
        steps_per_epoch=math.ceil(len(Y_train) / train_batch_size),
        epochs=10,
        callbacks=[evaluator],
        shuffle=False,
        initial_epoch=0
    )