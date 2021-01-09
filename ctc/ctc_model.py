#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/8 16:23
@File:          ctc_model.py
'''

from keras.layers import *

def CTC_Model(x, num_classes=26):

    x = Conv2D(16, 3, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    residual = Conv2D(64, 1, strides=2, use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(64, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    x = add([x, residual])

    residual = Conv2D(128, 1, strides=2, use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    x = add([x, residual])

    residual = Conv2D(364, 1, strides=2, use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = SeparableConv2D(364, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(364, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    x = add([x, residual])

    for i in range(8):
        residual = x

        x = Activation('relu')(x)
        x = SeparableConv2D(364, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(364, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(364, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = add([residual, x])

    residual = Conv2D(512, 1, strides=2, use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(364, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(512, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    x = add([x, residual])

    x = Permute((2, 1, 3))(x)
    x = Reshape((9, -1))(x)

    x = Bidirectional(CuDNNLSTM(256, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(512, return_sequences=True))(x)

    x = Dense(num_classes + 1)(x)

    return x