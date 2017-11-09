# coding: utf-8

import cfg
import re
import jieba
import pandas as pd
import datetime
import codecs


def load_csv(file_name):
    df = pd.read_csv(file_name, sep='\t|\n', encoding='utf8', header=None,
                     names=['id', 'head', 'content', 'label'], engine='python')
    return df


class Words:
    def __init__(self):
        self.user_dict_file = cfg.DATA_PATH + 'user_dict.txt'
        self.stop_words_file = cfg.DATA_PATH + 'stop_words.txt'
        self.train_file = cfg.DATA_PATH + 'train.tsv'  # 训练集原始文件
        self.test_file = cfg.DATA_PATH + 'evaluation_public.tsv'  # 测试集原始文件
        self.train_words_file = cfg.DATA_PATH + 'train_words.csv'  # 训练集分词结果
        self.test_words_file = cfg.DATA_PATH + 'test_words.csv'  # 测试集分词结果
        self.train_words_clean_file = cfg.DATA_PATH + 'train_words_clean.csv'  # 提取的训练集分词结果
        self.test_words_clean_file = cfg.DATA_PATH + 'test_words_clean.csv'  # 提取的测试集分词结果
        self.stop_words = set(' ')

    def set_stop_words(self):
        with open(self.stop_words_file, 'r') as f:
            for line in f:
                if line.strip() not in self.stop_words:
                    self.stop_words.add(line.strip())
        print('load stop words done.')
        print('stop words:', self.stop_words)

    def set_user_dict(self):
        jieba.load_userdict(self.user_dict_file)
        print('load user dict done.')

    def cut_train_file(self):
        jieba.enable_parallel(8)
        fw = codecs.open(self.train_words_file, 'w', encoding='utf8')
        with open(self.train_file, 'r') as f:
            for n, line in enumerate(f):
                line[1] = ' '.join([w for w in list(jieba.cut(line[1]))])
                line[2] = ' '.join([w for w in list(jieba.cut(line[2]))])
                if (n + 1) % 10000 == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print('%s, %s: %s done' % (time_str, self.train_file, (n + 1)))
                fw.write('%s\t%s\t%s\t%s\n' % (line[0], line[1], line[2], line[3]))
        print('save %s done.' % self.train_words_file)

    def cut_test_file(self):
        jieba.enable_parallel(8)
        fw = codecs.open(self.test_words_file, 'w', encoding='utf8')
        with open(self.test_file, 'r') as f:
            for n, line in enumerate(f):
                line = line.strip().split('\t')
                line[1] = ' '.join([w for w in list(jieba.cut(line[1]))])
                line[2] = ' '.join([w for w in list(jieba.cut(line[2]))])
                if (n + 1) % 10000 == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print('%s, %s: %s done' % (time_str, self.test_file, (n + 1)))
                fw.write('%s\t%s\t%s\t%s\n' % (line[0], line[1], line[2], 'POSITIVE'))
        print('save %s done.' % self.test_words_file)

    @staticmethod
    def clean(words_file, words_clean_file):
        pattern_one_char = re.compile(u' [A-Za-z]{1} ')  # 单英文字符
        pattern_2more_space = re.compile(' {2,}')  # 匹配2到无限个空格
        pattern = re.compile(u'[^\u4E00-\u9FA5A-Za-z]')  # 汉字和英文

        fw = codecs.open(words_clean_file, 'w', encoding='utf8')
        with open(words_file, 'r') as f:
            n = 0
            for n, line in enumerate(f):
                line = line.strip().split('\t')
                line[1] = re.sub(pattern, ' ', line[1])  # 保留 汉字 英文
                line[1] = re.sub(pattern_one_char, ' ', line[1])  # 过滤单个字符
                line[1] = re.sub(pattern_2more_space, ' ', line[1])  # 将多个连着的空格变为单个空格
                line[1] = line[1].lower()  # 大写转为小写
                line[2] = re.sub(pattern, ' ', line[2])
                line[2] = re.sub(pattern_one_char, ' ', line[2])
                line[2] = re.sub(pattern_2more_space, ' ', line[2])
                line[2] = line[2].lower()
                if (n + 1) % 10000 == 0:
                    print('clean %s %s done' % (words_file, n + 1))
                fw.write('%s\t%s\t%s\t%s\n' % (line[0], line[1], line[2], line[3]))
            print('clean %s %s done' % (words_file, n + 1))
            print('save %s done' % words_clean_file)

    def clean_words_file(self):
        self.clean(self.train_words_file, self.train_words_clean_file)
        self.clean(self.test_words_file, self.test_words_clean_file)

words = Words()