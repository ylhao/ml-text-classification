# coding: utf-8

import re
import time
import jieba
import cfg
import pandas as pd
import codecs
import threading
import datetime
import jieba.analyse
import jieba.analyse.tfidf

# 匹配 非汉字、非英文字符、非 \t、非 \n
# PATTERN_3 = re.compile(u'[^\u4E00-\u9FA5\u0041-\u005A\u0061-\u007A\t\n]')
PATTERN_3 = re.compile(u'[^\u4E00-\u9FA5\t\n]')
PATTERN_4 = re.compile(' {2,}')


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
        self.idf_file = cfg.DATA_PATH + 'idf.txt'
        self.stop_words = None
        self.words_dict = {}
        self.train_pos_tags = cfg.DATA_PATH + 'train_pos_tags.csv'
        self.train_neg_tags = cfg.DATA_PATH + 'train_neg_tags.csv'


    def idf(self, df_list):
        """
        计算所有词的反文档比例 idf
        :param df_list: DataFrame 列表
        """
        total_num = 0
        for df in df_list:
            total_num += df.shape[0]
            for line_num in range(df.shape[0]):
                words = []
                tf = {}
                words.extend(df.iloc[line_num]['headline'].split())
                words.extend(df.iloc[line_num]['content'].split())
                for word in words:
                    if word not in tf:  # 没有统计过该词的词频
                        tf[word] = 1
                        # 没有统计过该词的词频，也意味着没有分析该词在该文章中出现过没有
                        if word not in self.words_dict:
                            self.words_dict[word] = 1
                        else:
                            self.words_dict[word] += 1
                    else:
                        tf[word] += 1
        print('总文章数：', total_num)
        for word in self.words_dict:
            if self.words_dict[word] < 1:  #　只有1篇文章出现过该词
                self.words_dict[word] = 0
                continue
            self.words_dict[word] = float(total_num) / self.words_dict[word]

        with codecs.open(self.idf_file, 'w', encoding='utf-8') as f:
            for word in self.words_dict:
                if self.words_dict[word] != 0:
                    f.write('%s %f\n' %(word, self.words_dict[word]))

    def cut(self, df, words_file):
        """
        分词 去停用词
        :param df: train_df and test_df 
        :param words_file: 存放分词结果 
        """
        start = time.time()
        fw = codecs.open(words_file, 'w', encoding='utf-8')
        for line_num in range(df.shape[0]):

            df.iloc[line_num]['headline'] = re.sub(PATTERN_3, ' ', df.iloc[line_num]['headline'])
            df.iloc[line_num]['headline'] = re.sub('PATTERN_4', ' ', df.iloc[line_num]['headline'])
            df.iloc[line_num]['content'] = re.sub(PATTERN_3, ' ', df.iloc[line_num]['content'])
            df.iloc[line_num]['content'] = re.sub(PATTERN_4, ' ', df.iloc[line_num]['content'])

            # 分词 去停用词
            df.iloc[line_num]['headline'] = ' '.join([word for word in list(jieba.cut(df.iloc[line_num]['headline']))
                                                      if word not in self.stop_words])
            df.iloc[line_num]['content'] = ' '.join([word for word in list(jieba.cut(df.iloc[line_num]['content']))
                                                     if word not in self.stop_words])
            fw.write('%s\t%s\t%s\n' % (df.iloc[line_num]['id'],
                                       df.iloc[line_num]['headline'],
                                       df.iloc[line_num]['content']))

            # 进程监控
            if (line_num + 1) % 10000 == 0:
                time_str = datetime.datetime.now().isoformat()
                print('%s, %s: %s done' % (time_str, words_file, (line_num + 1)))
        fw.close()
        print(time.time() - start)

    def extract_tags(self, df):
        """
        提取关键词
        :param df: DataFrame 
        """
        analyse = MyTFIDF()
        fw1 = codecs.open(self.train_pos_tags, 'w', encoding='utf-8')
        fw2 = codecs.open(self.train_neg_tags, 'w', encoding='utf-8')
        for line_num in range(df.shape[0]):
            tags_headline = analyse.my_extract_tags(df.iloc[line_num]['headline'].split(), topK=3)
            while len(tags_headline) < 3:
                tags_headline.append('<PAD>')
            tags_content = analyse.my_extract_tags(df.iloc[line_num]['content'].split(), topK=56)
            while len(tags_content) < 56:
                tags_content.append('<PAD>')
            tags = tags_headline + tags_content
            if df.iloc[line_num]['label'] == 'POSITIVE':
                fw1.write(' '.join(tags) + '\n')
            else:
                fw2.write(' '.join(tags) + '\n')
        fw1.close()
        fw2.close()
            # print(df.iloc[line_num]['content'].split())
            # print(tags)


    def set_stop_words(self):
        """
        自定义停用词
        """
        self.stop_words = [line.strip('\n') for line in
                           codecs.open(self.stop_words_file, encoding='utf8').readlines()]
        print('stop words list:', self.stop_words)

    def set_user_dict(self):
        """
        自定义词典
        """
        jieba.load_userdict(self.user_dict_file)

    def set_idf_dict(self):
        """
        自定义idf词典
        """
        jieba.analyse.set_idf_path(self.idf_file)

    def run(self, df_list):
        self.set_stop_words()  # 设置停用词
        # self.set_user_dict()  # 自定义词典
        jieba.enable_parallel(4)  # 并行分词
        ts = [threading.Thread(target=self.cut, args=(df_list[0], self.train_words_file)),
              threading.Thread(target=self.cut, args=(df_list[1], self.test_words_file))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        # self.set_idf_dict()  # 导入自定义 idf 文件
        # self.idf(df_list)
        self.extract_tags(df_list[0])


class MyTFIDF(jieba.analyse.TFIDF):
    """
    自定义 TFIDF 类，用于 基于 tf-idf 提取关键词，并按原文顺序返回
    """
    def my_extract_tags(self, words, topK=20, withWeight=False, allowPOS=(), withFlag=False):
        """
        自定义的提取关键词的函数，按原顺序返回
            - words: 词列表 []
        """
        # print('words len:', len(words))
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
        # print('len tags', len(tags))
        tags1 = []
        for w in words:
            if w in tags and w not in tags1:
                tags1.append(w)
        # print('len tags1', len(tags1))
        return tags1
