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
        """
        :param filename_list: 文件名列表 
        """
        self.filename_list = filename_list

    def __iter__(self):
        for filename in self.filename_list:
            for line_num, line in enumerate(codecs.open(filename, encoding='utf8')):
                if (line_num + 1) % 50000 == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print('%s, %s done' % (time_str, line_num+1))
                line = line.strip().split('\t')
                words = []
                words.extend(line[1].split())  # head
                words.extend(line[2].split())  # content
                yield words


class W2VModelManager:
    """
    Word2Vec 模型管理器
    """
    def __init__(self):
        self.train_words_file = cfg.DATA_PATH + 'train_words.csv'
        self.test_words_file = cfg.DATA_PATH + 'test_words.csv'
        self.train_words_file_pos = cfg.DATA_PATH + 'train_words_pos.csv'
        self.model_name = cfg.MODEL_PATH + 'sg.w2v'  # sg=1

    def train_model(self):
        """
        train word2vec model
        :return: word2vec model
        """
        # sens = MySentences([self.train_words_file, self.test_words_file])
        sens = MySentences([self.train_words_file_pos])
        w2v = Word2Vec(sens, size=200, window=5, sg=1, min_count=10, workers=12, iter=20)
        w2v.save(self.model_name)
        print('save w2v model done')

    @staticmethod
    def load_model(model_name):
        """
        load word2vec model
        :param model_name: word2vec model name
        :return: word2vec model
        """
        return Word2Vec.load(cfg.MODEL_PATH + model_name)

"""
model = W2VModelManager.load_model('sg.w2v')
print(model.most_similar(positive=['经济', '消防'], negative=['救援'], topn=1))
print(model['经济'])
"""




