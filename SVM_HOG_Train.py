#!/usr/local/bin/python3.6
# -*- coding: utf-8 -*-
"""
__title__ = 'None'
__author__ = 'None'
__mtime__ = 'None'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import os
import myLogger
import numpy as np
import cv2 as cv
import random
import time


def load_data(pos_path, neg_path):
    pos_list = []
    neg_list = []

    pos = os.listdir(pos_path)
    neg = os.listdir(neg_path)

    for pos_p in pos:
        pos_list.append(os.path.join(pos_path, pos_p))
    for neg_p in neg:
        neg_list.append(os.path.join(neg_path, neg_p))

    return pos_list, neg_list


def load_train_samples(pos, neg):
    samples = []
    labels = []
    neg_list_for_corrct = []
    x_width = 64
    y_height = 128
    random.seed(1)
    for i in pos:
        img_matrix = cv.imread(i, cv.COLOR_BGR2GRAY)
        if img_matrix is not None:
            #取样本图片中间(128*64)部分
            if img_matrix.shape[0] > y_height and img_matrix.shape[1] > x_width:
                img_new = img_matrix[
                          (img_matrix.shape[0] - y_height) // 2:(img_matrix.shape[0] - y_height) // 2 + y_height,
                          (img_matrix.shape[1] - x_width) // 2:(img_matrix.shape[1] - x_width) // 2 + x_width]
                # img_new = cv.cvtColor(img_new, cv.COLOR_BGR2GRAY)
                samples.append(img_new)
                labels.append(1.)

    for i in neg:
        img_matrix = cv.imread(i, cv.COLOR_BGR2GRAY)
        if img_matrix is not None:
            neg_list_for_corrct.append(img_matrix)
            #随机获取10份(128*64)的图片：扩大训练数据集10倍
            for j in range(10):
                x = int(random.random() * (img_matrix.shape[1] - x_width))
                y = int(random.random() * (img_matrix.shape[0] - y_height))
                img_new = img_matrix[y:y + y_height, x:x + x_width]
                # img_new = cv.cvtColor(img_new,cv.COLOR_BGR2GRAY)
                samples.append(img_new)
                labels.append(-1.)

    labels_len = len(labels)
    labels = np.int32(labels)
    labels = np.resize(labels, (labels_len,))

    return samples, neg_list_for_corrct, labels


def extract_hog(samples):
    train = []
    colors_channel = 3
    # hog = cv.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    hog = cv.HOGDescriptor()  #默认参数为训练128*64的数据集的参数
    for img in samples:
        if img.shape == (128, 64, colors_channel):
            descriptors = hog.compute(img)
            train.append(descriptors)
    train = np.float32(train)
    train = np.resize(train, (len(samples), 3780, 1))

    return train

def get_svm_detector(svm):
    '''
    导出可以用于cv2.HOGDescriptor()的SVM检测器，实质上是训练好的SVM的支持向量和rho参数组成的列表
    :param svm: 训练好的SVM分类器
    :return: SVM的支持向量和rho参数组成的列表，可用作cv2.HOGDescriptor()的SVM检测器
    '''
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    sv = np.transpose(sv)
    return np.append(sv, [[-rho]], 0)


def train_SVM(train, labels, Logger):
    svm = cv.ml.SVM_create()
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)  # for EPSILON_SVR, epsilon in loss function?
    svm.setC(0.01)  # From paper, soft classifier
    svm.setType(cv.ml.SVM_EPS_SVR)  # C_SVC # EPSILON_SVR # may be also NU_SVR # do regression task

    Logger.writeLog('Starting training svm...', level='info')
    time_start = time.time()
    svm.train(train, cv.ml.ROW_SAMPLE, labels)
    time_use = time.time() - time_start
    Logger.writeLog('Training done,use time:{}'.format(time_use), level='info')

    # return SVM检测器(直接用于HOG计算)，SVM分类器
    return get_svm_detector(svm), svm


def train_correction(hog, samples, labels, neg_list, svm, Logger, train_correction_count):
    Logger.writeLog('{}.th correction is training...'.format(train_correction_count), level='info')
    start_time = time.time()
    count = 0
    labels = labels.tolist()
    for img in neg_list:
        rects, _ = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
        for (x, y, w, h) in rects:
            img_wrong = img[y:y + h, x:x + w]
            samples.append(cv.resize(img_wrong, (64, 128)))
            labels.append(-1)
            count += 1
    train = extract_hog(samples)

    labels_len = len(labels)
    labels = np.int32(labels)
    labels = np.resize(labels, (labels_len,))
    svm.train(train, cv.ml.ROW_SAMPLE, labels)

    Logger.writeLog('{}.th correction train is done,use time:{},wrong samples:{}'.format(train_correction_count,
                                                                                         int(time.time() - start_time),
                                                                                         count),
                    level='info')
    return get_svm_detector(svm), svm, samples, labels, count


if __name__ == '__main__':

    pos_path = r'F:/OpenCV/SVM_HOG_DATA/Train/pos_set'
    neg_path = r'F:/OpenCV/SVM_HOG_DATA/Train/neg_set'

    Logger = myLogger.LogHelper()
    Logger.writeLog("Program runing...", level='info')
    Program_start = time.time()

    #获取正负样本图片路径
    pos, neg = load_data(pos_path,neg_path)
    Logger.writeLog('load pos samples:' + str(len(pos)), level='info')
    Logger.writeLog('load neg samples:' + str(len(neg)), level='info')
    #加载正负样本数据(全样本数据，负样本图片库，全样本对应标签)
    samples, neg_list_for_corrct, labels = load_train_samples(pos, neg)

    #计算正负样本的HOG特征
    train = extract_hog(samples)
    Logger.writeLog('Size of feature vectors of samples:{}'.format(train.shape), level='info')
    Logger.writeLog('Size of labels of samples:{}'.format(labels.shape), level='info')

    #训练
    SVM_detector, svm = train_SVM(train, labels, Logger)
    train_correction_count = 0  #修正训练计数

    #用训练好的检测器检测负样本图片库，将错误识别加入全样本数据集重复训练
    while True:
        hog = cv.HOGDescriptor()
        hog.setSVMDetector(SVM_detector)
        train_correction_count += 1
        #参数：hog对象，所有样本，标签，原始负样本，svm分类器，日志对象，训练计数
        #返回：svm检测器，cvm分类器，所有样本，对应标签，错误样本数
        SVM_detector, svm, samples, labels, wrong_count = train_correction(hog,
                                                                           samples,
                                                                           labels,
                                                                           neg_list_for_corrct,
                                                                           svm,
                                                                           Logger,
                                                                           train_correction_count)

        # if wrong_count <= 100 or train_correction_count >= 10:
        if wrong_count <= 100:
            break

    hog.save('myHogDector.bin')
    Logger.writeLog('Program is done...,use time:{}'.format(time.time() - Program_start),
                    level='info')
