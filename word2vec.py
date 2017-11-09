# coding: utf-8

import cfg
from gensim.models import Word2Vec
import datetime
from data_helpers import load_csv


class MySentences(object):
    def __init__(self, df_list=None):
        train_words_clean_file = cfg.DATA_PATH + 'train_words_clean.csv'
        test_words_clean_file = cfg.DATA_PATH + 'test_words_clean.csv'
        if not df_list:
            self.df_list = [load_csv(train_words_clean_file), load_csv(test_words_clean_file)]

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
    def __init__(self):
        self.model_name = cfg.MODEL_PATH + 'sg.w2v'

    def train_model(self):
        sens = MySentences()
        w2v = Word2Vec(sens, size=200, window=5, sg=1, min_count=3, workers=12, iter=20)
        w2v.save(self.model_name)
        print('save w2v model done')

    def load_model(self):
        return Word2Vec.load(self.model_name)

    def model_test(self):
        model = self.load_model()
        words = ['经济', '酒', '在', '的', '再', '年', '分', '社会', '足球', 'CEO',
                '股票', '手机', '苹果', '油桶', '油价', '美女', '失恋', '消防', '共产党',
                 '他', '电话', '联系电话', '编辑']
        for word in words:
            for w in (model.most_similar(positive=[word], topn=10)):
                print(w)
            print('-'*60)

w2vm = W2VModelManager()