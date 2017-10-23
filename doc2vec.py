# coding: utf-8

import cfg
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import datetime
from data_helpers import load_csv


class DocList(object):
    """
    文档迭代器
    """
    def __init__(self, df_list=None):
        train_words_file = cfg.DATA_PATH + 'train_words.csv'
        test_words_file = cfg.DATA_PATH + 'test_words.csv'
        if not df_list:
            self.df_list = [load_csv(train_words_file), load_csv(test_words_file)]

    def __iter__(self):
        tag = ['Train', 'Test']
        for i, df in enumerate(self.df_list):
            for line_num in range(df.shape[0]):
                if (line_num + 1) % 50000 == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print('%s, iter %s done' % (time_str, line_num + 1))
                words = []
                try:
                    words.extend(df.iloc[line_num]['head'].split())
                except:
                    print("line num %s head is nan" % line_num)
                try:
                    words.extend(df.iloc[line_num]['content'].split())
                except:
                    print("line num %s content is nan" % line_num)
                yield LabeledSentence(words, ['%s_%s' % (tag[i], words)])


class D2VModelManager:
    """
    Doc2Vec 模型管理器
    """
    def __init__(self):
        self.dm_model_name = cfg.MODEL_PATH + 'dm.d2v'

    def train_model(self):
        """
        训练 doc2vec model
        :return: doc2vec model
        """
        doc_l = DocList()
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


