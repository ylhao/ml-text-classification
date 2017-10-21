# coding: utf-8

import cfg
import re
import jieba
import pandas as pd
import threading
import datetime


PATTERN_1 = re.compile("([^\u4E00-\u9FA5a-zA-Z0-9\t\n])", re.U)
# PATTERN_1 = re.compile(u'[^\u4E00-\u9FA5\u0041-\u005A\u0061-\u007A\t\n]')
PATTERN_2 = re.compile(u'[^\u4E00-\u9FA5\t\n]')
PATTERN_3 = re.compile(' {2,}')  # 匹配 2到无限个空格


def load_csv(file_name):
    """
    加载 csv 文件
    """
    df = pd.read_csv(cfg.DATA_PATH + file_name, sep='\t|\n', encoding='utf8', header=None,
                     names=['id', 'head', 'content', 'label'], engine='python')
    return df


class DataHelper:
    def __init__(self):
        self.train_df = load_csv('train.tsv')
        self.test_df = load_csv('evaluation_public.tsv')
        self.test_df['label'] = 'POSITIVE'

    def clean_df(self):

        # train_df 删除有空值的行和重复行
        self.df_summary(self.train_df)
        self.train_df = self.train_df.dropna(axis=0, how='any')
        self.train_df = self.train_df.drop_duplicates(['head', 'content'])
        self.df_summary(self.train_df)

        # test_df 删除有空值的行
        self.df_summary(self.test_df)
        self.test_df = self.test_df.dropna(axis=0, how='any')
        self.df_summary(self.test_df)
        return self.train_df, self.test_df

    @staticmethod
    def df_summary(df):
        labels = ['head', 'content']
        print(df.info())
        print('-' * 120)
        for label in labels:
            print('%s count:' % label, df[label].count())
            print('%s unique:' % label,  df[label].unique().shape[0])
        print('#' * 120)


class Words:
    """
    分词 提取关键词
    """
    def __init__(self):
        self.user_dict_file = cfg.DATA_PATH + 'user_dict.txt'  # 分词用的自定义词典
        self.stop_words_file = cfg.DATA_PATH + 'stop_words.txt'  # 停词表
        self.train_words_file = cfg.DATA_PATH + 'train_words.csv'  # 训练集分词结果
        self.test_words_file = cfg.DATA_PATH + 'test_words.csv'  # 测试集分词结果
        self.stop_words = {
            ' ': 1,
        }

    def set_stop_words(self):
        with open(self.stop_words_file, 'r') as f:
            for line in f:
                self.stop_words[line.strip()] = 1
        print('stop words:', self.stop_words)

    def set_user_dict(self):
        jieba.load_userdict(self.user_dict_file)
        print('load user dict done')

    def cut(self, df, words_file):
        """
        分词，去停用词
        """
        for n in range(df.shape[0]):
            # head
            df.iloc[n]['head'] = re.sub(PATTERN_2, ' ', df.iloc[n]['head'])
            df.iloc[n]['head'] = re.sub(PATTERN_3, ' ', df.iloc[n]['head'])
            df.iloc[n]['head'] = ' '.join([w for w in list(jieba.cut(df.iloc[n]['head']))
                                           if w not in self.stop_words])
            # content
            df.iloc[n]['content'] = re.sub(PATTERN_2, ' ', df.iloc[n]['content'])
            df.iloc[n]['content'] = re.sub(PATTERN_3, ' ', df.iloc[n]['content'])
            df.iloc[n]['content'] = ' '.join([w for w in list(jieba.cut(df.iloc[n]['content']))
                                                     if w not in self.stop_words])
            # time
            if (n + 1) % 10000 == 0:
                time_str = datetime.datetime.now().isoformat()
                print('%s, %s: %s done' % (time_str, words_file, (n + 1)))
        df.to_csv(words_file, sep='\t', header=None, index=None, encoding='utf8')
        print('save %s done' % words_file)

    def cut_sens(self, df_list):
        print('分词 去停用词')
        ts = [threading.Thread(target=self.cut, args=(df_list[0], self.train_words_file)),
              threading.Thread(target=self.cut, args=(df_list[1], self.test_words_file))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()




