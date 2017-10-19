# coding: utf-8

import re
import jieba
import cfg
import pandas as pd
import codecs
import threading
import datetime
import jieba.analyse
import jieba.analyse.tfidf
import numpy as np
from scipy.stats import chi2


# 匹配 非汉字、非英文字符、非 \t、非 \n
PATTERN_1 = re.compile(u'[^\u4E00-\u9FA5\u0041-\u005A\u0061-\u007A\t\n]')
# 匹配 非汉字、非 \t、非 \n
PATTERN_2 = re.compile(u'[^\u4E00-\u9FA5\t\n]')
# 匹配 2到无限个空格
PATTERN_3 = re.compile(' {2,}')


def load_csv(file_name):
    df = pd.read_csv(file_name, sep='\t|\n', encoding='utf8', header=None,
                     names=['id', 'head', 'content', 'label'], engine='python')
    return df


class DataHelper:
    def __init__(self):
        self.train_df = load_csv(cfg.DATA_PATH + 'train.tsv')
        self.test_df = load_csv(cfg.DATA_PATH + 'evaluation_public.tsv')
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

    @staticmethod
    def df_summary(df):
        labels = ['head', 'content']
        print(df.info())
        print('-' * 120)
        for label in labels:
            print('%s count:' % label, df[label].count())
            print('%s unique:' % label,  df[label].unique().shape[0])
        print('#' * 120)

    def run(self):
        self.clean_df()
        return self.train_df, self.test_df


class Doc2Words:
    """
    分词 idf chi 
    """
    def __init__(self):
        self.user_dict_file = cfg.DATA_PATH + 'user_dict.txt'  # 分词用的自定义词典
        self.stop_words_file = cfg.DATA_PATH + 'stop_words.txt'  # 停词表
        self.idf_file = cfg.DATA_PATH + 'idf.txt'  # idf文档
        self.train_words_file = cfg.DATA_PATH + 'train_words.csv'  # 训练集分词结果
        self.test_words_file = cfg.DATA_PATH + 'test_words.csv'  # 测试集分词结果
        self.train_words_pos_file = cfg.DATA_PATH + 'train_words_pos_file.csv'  # 训练集正样本分词结果
        self.train_tags_pos_file = cfg.DATA_PATH + 'train_tags_pos.csv'  # 训练集正样本关键词提取结果
        self.train_tags_neg_file = cfg.DATA_PATH + 'train_tags_neg.csv'  # 训练集负样本关键词提取结果
        self.test_tags_file = cfg.DATA_PATH + 'test_tags.csv'  # 测试集关键词提取结果
        self.stop_words = [' ']
        self.words_idf = {}  # 统计该词的 idf
        self.pos = {}
        self.neg = {}

    def set_stop_words(self):
        self.stop_words.extend([line.strip() for line in open(self.stop_words_file).readlines()])
        self.stop_words.remove('')
        print('stop words:', self.stop_words)

    def set_user_dict(self):
        jieba.load_userdict(self.user_dict_file)
        print('load user dict done')

    def set_idf_dict(self):
        jieba.analyse.set_idf_path(self.idf_file)
        print('set user idf file done')

    def cut(self, df, words_file):
        """
        分词 去停用词
        :param df: 数据框 
        :param words_file: 存放分词结果的文件名 
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

    def idf(self, df_list):
        """
        :param df_list: 数据框列表
        """
        if not df_list:
            df_list = [load_csv(self.train_words_file), load_csv(self.test_words_file)]
        total_num = 0  # 总文章数
        for df in df_list:
            total_num += df.shape[0]
            for n in range(df.shape[0]):
                done = {}
                words = []
                words.extend(df.iloc[n]['head'].split())
                words.extend(df.iloc[n]['content'].split())
                for w in words:
                    if w not in done:  # 没有统计过该词的词频
                        done[w] = True
                        # 没有统计过该词的词频，也意味着没有分析 该词 在 该文章 中出现过没有
                        if w not in self.words_idf:
                            self.words_idf[w] = 1
                        else:
                            self.words_idf[w] += 1
                    else:
                        continue
        print('总文章数：', total_num)
        for w in self.words_idf:
            self.words_idf[w] = float(total_num) / self.words_idf[w]
        with codecs.open(self.idf_file, 'w', encoding='utf-8') as f:
            for w in self.words_idf:
                if self.words_idf[w] != 0:
                    f.write('%s %f\n' % (w, self.words_idf[w]))
        print('save %s done' % self.idf_file)

    def extract_train_tags(self, train_df=None, head_topK=3, content_topK=56):
        """
        提取关键词
        :param train_df: 训练集数据框
        :param head_topK: 要从标题中提取的关键词数
        :param content_topK: 要从内容中提取的关键词数
        """
        if not train_df:
            train_df = load_csv(self.train_words_file)
        analyse = MyTFIDF()
        fw_pos = codecs.open(self.train_tags_pos_file, 'w', encoding='utf-8')
        fw_neg = codecs.open(self.train_tags_neg_file, 'w', encoding='utf-8')
        for n in range(train_df.shape[0]):
            tags_head = analyse.my_extract_tags(train_df.iloc[n]['head'].split(), topK=head_topK)
            while len(tags_head) < head_topK:
                tags_head.append('<PAD>')
            tags_content = analyse.my_extract_tags(train_df.iloc[n]['content'].split(), topK=content_topK)
            while len(tags_content) < content_topK:
                tags_content.append('<PAD>')
            tags = tags_head + tags_content
            if train_df.iloc[n]['label'] == 'POSITIVE':
                fw_pos.write(' '.join(tags) + '\n')
            else:
                fw_neg.write(' '.join(tags) + '\n')
        fw_pos.close()
        fw_neg.close()
        print('extract train tags done')

    def chi(self, train_df=None):
        """
        CHI
        """
        if not train_df:
            train_df = load_csv(self.train_words_file)

        for n in range(train_df.shape[0]):
            words = []
            words.extend(train_df.iloc[n]['head'])
            words.extend(train_df.iloc[n]['content'])
            for w in words:
                done = {}  # 该行已经统计过的词
                if train_df.iloc[n]['label'] == 'POSITIVE':
                    if w not in done:
                        done[w] = True
                        if w not in self.pos:
                            self.pos[w] = 1
                        else:
                            self.pos[w] += 1
                    else:
                        continue
                if train_df.iloc[n]['label'] == 'NEGATIVE':
                    if w not in done:
                        done[w] = True
                        if w not in self.pos:
                            self.pos[w] = 1
                        else:
                            self.pos[w] += 1
                    else:
                        continue

    def extract_pos_sample(self):
        train_df = load_csv(self.train_words_file)
        train_df[train_df['label'] == 'POSITIVE'].to_csv(
            self.train_tags_pos_file,
            sep='\t', header=None, index=None, encoding='utf8'
        )

    def run(self, df_list):

        self.set_stop_words()  # 停用词
        print('-' * 120)

        # self.set_user_dict()  # 自定义词典
        # print('-' * 120)

        # jieba.enable_parallel(4)  # 并行分词
        # ts = [threading.Thread(target=self.cut, args=(df_list[0], self.train_words_file)),
        #       threading.Thread(target=self.cut, args=(df_list[1], self.test_words_file))]
        # for t in ts:
        #     t.start()
        # for t in ts:
        #     t.join()
        # print('-' * 120)

        # self.idf(df_list)
        # print('-' * 120)
        #
        # self.set_idf_dict()
        # print('-' * 120)

        self.extract_train_tags(head_topK=6, content_topK=250)
        print('-' * 120)

        # self.extract_pos_sample()
        print('-' * 120)


class MyTFIDF(jieba.analyse.TFIDF):

    def my_extract_tags(self, words, topK=20, withWeight=False, allowPOS=(), withFlag=False):
        """
        自定义基于 tf-idf 提取关键词的函数 并按词的原顺序返回
        """
        freq = {}
        for w in words:
            if allowPOS:
                if w.flag not in allowPOS:
                    continue
                elif not withFlag:
                    w = w.word
            wc = w.word if allowPOS and withFlag else w
            if len(wc.strip()) < 2 or wc.lower() in self.stop_words:
                continue
            freq[w] = freq.get(w, 0.0) + 1.0
        total = sum(freq.values())
        for k in freq:
            kw = k.word if allowPOS and withFlag else k
            freq[k] *= self.idf_freq.get(kw, self.median_idf) / total
        tags = sorted(freq, key=freq.__getitem__, reverse=True)
        tags = tuple(tags[:topK])
        tags1 = []
        for w in words:
            if w in tags and w not in tags1:
                tags1.append(w)
        return tags1
