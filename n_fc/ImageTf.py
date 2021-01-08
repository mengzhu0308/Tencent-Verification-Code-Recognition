#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/4 18:43
@File:          ImageTf.py
'''

import cv2

class ImageTf:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, image):
        image = cv2.resize(image, tuple(reversed(self.image_size)))
        return image / 255