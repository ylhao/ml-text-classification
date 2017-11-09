# coding utf-8

from data_helpers import words
from word2vec import w2vm
from data_split import pre_train, pre_evl

pre_train()
pre_evl()
words.cut_train_file()
words.cut_test_file()
words.clean_words_file()
w2v = w2vm.train_model()