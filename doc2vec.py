# coding: utf-8

import codecs
import cfg
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import datetime

class DocList(object):
    """
    文档迭代器
    """
    def __init__(self, filename_list):
        self.filename_list = filename_list

    def __iter__(self):
        for filename in self.filename_list:
            for line_num, line in enumerate(codecs.open(filename, encoding='utf8')):
                if (line_num + 1) % 50000 == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print('%s, %s done' % (time_str, line_num + 1))
                line = line.split('\t')
                words = []
                words.extend(line[1].split())
                words.extend(line[2].split())
                if 'train_' in filename:
                    yield LabeledSentence(words, ['TRAIN_%s' % (line[0])])
                elif 'test_' in filename:
                    yield LabeledSentence(words, ['TEST_%s' % line[0]])
                else:
                    print('文件名有误')


class D2VModelManager:
    """
    Doc2Vec模型管理器
    """
    def __init__(self):
        self.train_words_file = cfg.DATA_PATH + 'train_words.csv'
        self.test_words_file = cfg.DATA_PATH + 'test_words.csv'
        self.dm_model_name = cfg.MODEL_PATH + 'dm.d2v'

    def train_model(self):
        """
        训练 doc2vec model
        :return: doc2vec model
        """
        doc_l = DocList([self.train_words_file, self.test_words_file])
        d2v = Doc2Vec(dm=1, size=300, negative=5, hs=1, sample=1e-5,
                      window=10, min_count=5, workers=12, alpha=0.025, min_alpha=0.025)
        d2v.build_vocab(doc_l)
        for epoch in range(10):
            d2v.train(doc_l, total_examples=d2v.corpus_count, epochs=1)
            d2v.alpha -= 0.002
            d2v.min_alpha = d2v.alpha
            print('%s epoch done' % (epoch + 1))
        d2v.save(self.dm_model_name)
        print('save d2v model done')
        return d2v

    @staticmethod
    def load_model(model_name):
        """
        加载训练完成的 doc2vec model
        :param model_name: 模型名
        :return: doc2vec model
        """
        return Doc2Vec.load(cfg.MODEL_PATH + model_name)


