#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-
"""
__title__ = 'None'
__author__ = 'None'
__mtime__ = 'None'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""

import cv2
import time
import numpy as np
import os

def test_svm(hog):
    test_list = []
    test = os.listdir(r'F:/OpenCV/SVM_HOG_DATA/Test')
    for i in test:
        test_list.append(os.path.join(r'F:/OpenCV/SVM_HOG_DATA/Test', i))
    i = 0
    for f in test_list:
        i += 1
        print(i)
        img = cv2.imread(f,cv2.COLOR_BGR2GRAY)
        rects, _ = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
        for (x,y,w,h) in rects:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('{}'.format(i),img)
        if i>=10:
            break
    cv2.waitKey()

def test_svm_vidio(hog):
    cap = cv2.VideoCapture(0)
    while True:
        img = cap.read()[1]
        time_1 = time.time()
        rects, wei = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
        print(time.time() - time_1)
        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow('a', img)
        if cv2.waitKey(33) == 27:
            break
    cv2.destroyAllWindows()


hog = cv2.HOGDescriptor()
hog.load('myHogDector.bin')
#官方自带的检测器
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#两种测试方式：1.测试数据集  2.视频测试
test_svm(hog)
# test_svm_vidio(hog)