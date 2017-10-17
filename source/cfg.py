# coding: utf-8

import os

DATA_PATH = '/home/ylhao/PycharmProjects/360/data/'
MODEL_PATH = '/home/ylhao/PycharmProjects/360/model/'
RESULT_PATH = '/home/ylhao/PycharmProjects/360/result/'
CLASSIFIER_PATH = '/home/ylhao/PycharmProjects/360/classifier/'
LOG_PATH = '/home/ylhao/PycharmProjects/360/logs/'

def mkdir(path):
    """
    创建目录
    """
    if not os.path.exists(path):
        os.makedirs(path)


mkdir(DATA_PATH)
mkdir(MODEL_PATH)
mkdir(RESULT_PATH)
mkdir(CLASSIFIER_PATH)
mkdir(LOG_PATH)
