#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/4 17:57
@File:          ocr_model.py
'''

from keras.layers import *

def OCR_Model(x, num_classes=26, seq_len=4):
    x = Conv2D(16, 3, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, 3, use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    residual = Conv2D(64, 1, use_bias=False)(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(64, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=3, strides=1, padding='same')(x)
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

    x = SeparableConv2D(768, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(1024, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    xs = [Dense(num_classes)(x) for _ in range(seq_len)]
    x = concatenate(xs)

    return x