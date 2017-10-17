# coding: utf-8

import codecs
import cfg
from gensim.models import Word2Vec
import time
import datetime


class MySentences(object):
    """
    Word2Vec 使用的文档迭代器
    """
    def __init__(self, filename_list):
        self.filename_list = filename_list

    def __iter__(self):
        for filename in self.filename_list:
            for line_num, line in enumerate(codecs.open(filename, encoding='utf8')):
                if (line_num + 1) % 50000 == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print('%s, %s done' % (time_str, line_num+1))
                line = line.split('\t')
                words = []
                words.extend(line[1].split())
                words.extend(line[2].split())
                yield words


class W2VModelManager:
    """
    Word2Vec 模型管理器
    """
    def __init__(self):
        self.train_words_file = cfg.DATA_PATH + 'train_words.csv'
        self.test_words_file = cfg.DATA_PATH + 'test_words.csv'
        self.model_name = cfg.MODEL_PATH + 'sg.w2v'  # sg=1

    def train_model(self):
        """
        训练 word2vec model
        :return: word2vec model
        """
        sens = MySentences([self.train_words_file, self.test_words_file])
        w2v = Word2Vec(sens, size=200, window=5, sg=1, min_count=10, workers=12, iter=20)
        w2v.save(self.model_name)
        print('save w2v model done')

    @staticmethod
    def load_model(model_name):
        """
        加载训练完成的 word2vec model
        :param model_name: 模型名
        :return: word2vec model
        """
        return Word2Vec.load(cfg.MODEL_PATH + model_name)


