# coding: utf-8

import os

DATA_PATH = '../data/'
MODEL_PATH = '../model/'
RESULT_PATH = '../result/'
CLASSIFIER_PATH = '../classifier/'
LOG_PATH_CNN = '../logs/cnn/'
LOG_PATH_TEXTCNN = '../logs/text_cnn/'


def mkdir(path):
    """
    make directory
    """
    if not os.path.exists(path):
        os.makedirs(path)


mkdir(DATA_PATH)
mkdir(MODEL_PATH)
mkdir(RESULT_PATH)
mkdir(CLASSIFIER_PATH)
mkdir(LOG_PATH_CNN)
mkdir(LOG_PATH_TEXTCNN)