# coding: utf-8

import re
import time
import jieba
import cfg
import pandas as pd
import codecs
import threading
import datetime

# 匹配 非汉字、非英文字符、非 \t、非 \n
PATTERN_3 = re.compile(u'[^\u4E00-\u9FA5\u0041-\u005A\u0061-\u007A\t\n]')


class DataHelper:
    """
    预处理
    """
    def __init__(self):
        # file
        self.train_file = cfg.DATA_PATH + 'train.tsv'
        self.test_file = cfg.DATA_PATH + 'evaluation_public.tsv'
        # df
        self.train_df = pd.read_csv(self.train_file, sep='\t|\n', encoding='utf8', header=None,
                                    names=['id', 'headline', 'content', 'label'], engine='python')
        self.test_df = pd.read_csv(self.test_file, sep='\t|\n', encoding='utf8',
                                   header=None, names=['id', 'headline', 'content'], engine='python')

    def clean_train_df(self):
        """
        训练集数据清洗，删除有 NaN 值的行，删除重复行
        """
        self.df_summary(self.train_df)
        self.train_df = self.train_df.dropna(axis=0, how='any')
        self.train_df = self.train_df.drop_duplicates(['headline', 'content'])
        self.df_summary(self.train_df)

    def clean_test_df(self):
        """
        测试集数据清洗，删除有 NaN 值的行
        """
        self.df_summary(self.test_df)
        self.test_df = self.test_df.dropna(axis=0, how='any')
        self.df_summary(self.test_df)

    @staticmethod
    def df_summary(df):
        print(df.info())
        print('-'*120)
        print(df.describe())
        print('#' * 120)

    def run(self):
        """
        :return: train_df, test_df 
        """
        self.clean_train_df()
        self.clean_test_df()
        return self.train_df, self.test_df


class Doc2Words:
    """
    分词
    """
    def __init__(self):
        self.user_dict_file = cfg.DATA_PATH + 'user_dict.txt'
        self.stop_words_file = cfg.DATA_PATH + 'stop_words.txt'
        self.train_words_file = cfg.DATA_PATH + 'train_words.csv'
        self.test_words_file = cfg.DATA_PATH + 'test_words.csv'
        self.stop_words = None

    def cut(self, df, words_file):
        """
        分词
        :param df: train_df and test_df 
        :param words_file: train_words_file and test_words_file 
        """
        start = time.time()
        fw = codecs.open(words_file, 'w', encoding='utf-8')
        for line_num in range(df.shape[0]):
            headline_words = [word for word in list(jieba.cut(re.sub(PATTERN_3, ' ', df.iloc[line_num]['headline'])))
                              if word not in self.stop_words]
            content_words = [word for word in list(jieba.cut(re.sub(PATTERN_3, ' ', df.iloc[line_num]['content'])))
                              if word not in self.stop_words]
            if (line_num + 1) % 10000 == 0:
                time_str = datetime.datetime.now().isoformat()
                print('%s, %s: %s done' % (time_str, words_file, (line_num + 1)))
            fw.write('%s\t%s\t%s\n' % (df.iloc[line_num]['id'], ' '.join(headline_words), ' '.join(content_words)))
        fw.close()
        print(time.time() - start)

    def set_stop_words(self):
        self.stop_words = [line.strip('\n') for line in
                           codecs.open(self.stop_words_file, encoding='utf8').readlines()]
        print(self.stop_words)

    def set_user_dict(self):
        jieba.load_userdict(self.user_dict_file)

    def run(self, df_list):
        self.set_stop_words()  # 设置停用词
        # self.set_user_dict()  # 自定义词典
        jieba.enable_parallel(12)
        ts = [threading.Thread(target=self.cut, args=(df_list[0], self.train_words_file)),
              threading.Thread(target=self.cut, args=(df_list[1], self.test_words_file))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
