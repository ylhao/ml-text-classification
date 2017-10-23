# coding: utf-8

import cfg
from gensim.models import Word2Vec
import datetime
from data_helpers import load_csv


class MySentences(object):
    """
    Word2Vec 使用的文档迭代器
    """
    def __init__(self, df_list=None):
        """
        :param filename_list: 文件名列表 
        """
        train_words_file = cfg.DATA_PATH + 'train_new.csv'
        test_words_file = cfg.DATA_PATH + 'test_new.csv'
        if not df_list:
            self.df_list = [load_csv(train_words_file), load_csv(test_words_file)]

    def __iter__(self):
        for df in self.df_list:
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
                yield words


class W2VModelManager:
    """
    Word2Vec 模型管理器
    """
    def __init__(self):
        self.model_name = cfg.MODEL_PATH + 'sg.w2v'  # sg=1

    def train_model(self):
        """
        train word2vec model
        :return: word2vec model
        """
        sens = MySentences()
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
        return Word2Vec.load(model_name)

    @staticmethod
    def model_test(model_name):
        model = W2VModelManager.load_model(model_name)
        for w in (model.most_similar(positive=['经济'], topn=10)):
            print(w)
        print('-'*60)
        for w in (model.most_similar(positive=['{'], topn=10)):
            print(w)
        print('-'*60)
        for w in (model.most_similar(positive=['酒'], topn=10)):
            print(w)
        print('-'*60)
        for w in (model.most_similar(positive=['年'], topn=10)):
            print(w)
        print('-'*60)
        for w in (model.most_similar(positive=['点'], topn=10)):
            print(w)
        print('-'*60)
        for w in (model.most_similar(positive=['分'], topn=10)):
            print(w)
        print('-'*60)
        for w in (model.most_similar(positive=['月'], topn=10)):
            print(w)
        print('-'*60)
        for w in (model.most_similar(positive=['法律'], topn=10)):
            print(w)
        print('-'*60)
        for w in (model.most_similar(positive=['社会'], topn=10)):
            print(w)
        print('-'*60)
        for w in (model.most_similar(positive=['股票'], topn=10)):
            print(w)
        print('-'*60)
        for w in (model.most_similar(positive=['手机'], topn=10)):
            print(w)
        print('-'*60)
        for w in (model.most_similar(positive=['交通'], topn=10)):
            print(w)
        print('-'*60)
        print(model['经济'])
        print(model['的'])






