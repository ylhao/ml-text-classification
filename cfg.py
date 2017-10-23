# coding: utf-8

import re
import os


ROOT_PATH = re.match(r'\S+360',  os.getcwd()).group()

DATA_PATH = ROOT_PATH + '/data/'  # 所有数据
MODEL_PATH = ROOT_PATH + '/model/'  # 所有模型
RESULT_PATH = ROOT_PATH + '/result/'  # 所有结果
CLASSIFIER_PATH = ROOT_PATH + '/classifier/'  # 机器学习算法得到的所有分类器
TEXT_CNN_PATH = ROOT_PATH + '/model/text-cnn/'


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


mkdir(DATA_PATH)
mkdir(MODEL_PATH)
mkdir(RESULT_PATH)
mkdir(CLASSIFIER_PATH)
mkdir(TEXT_CNN_PATH)
