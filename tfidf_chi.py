# coding: utf-8

# coding: utf-8

import cfg
import codecs
import datetime
from data_helpers import load_csv
from operator import itemgetter
import threading
import math


class IDF:
    """
    计算每个词的反文档比例
    """

    def __init__(self, df_list=None):

        self.idf_file = 'idf.txt'  # idf文档
        self.idf = {}  # 统计该词的 idf

        self.df_list = df_list
        if not df_list:
            self.df_list = [load_csv('train_words.csv'), load_csv('test_words.csv')]

    def compute_idf(self):
        total_num = 0  # 文章总数
        for df in self.df_list:
            total_num += df.shape[0]
            for n in range(df.shape[0]):
                done = {}
                words = []
                try:
                    words.extend(df.iloc[n]['head'].split())
                except:
                    print('%s head is nan' % n)
                try:
                    words.extend(df.iloc[n]['content'].split())
                except:
                    print('%s content is nan' % n)
                for w in words:
                    if w not in done:  # 没有统计过该词的词频
                        done[w] = True
                        # 没有统计过该词的词频，也意味着没有分析 该词 在 该文章 中出现过没有
                        if w not in self.idf:
                            self.idf[w] = 1
                        else:
                            self.idf[w] += 1
                    else:
                        continue
        print('总文章数：', total_num)
        for w in self.idf:
            self.idf[w] = math.log(float(total_num) / (self.idf[w] + 1))

        fw = codecs.open(self.idf_file, 'w', encoding='utf-8')
        for w in self.idf:
            fw.write('%s %f\n' % (w, self.idf[w]))
        print('save %s done' % self.idf_file)


class CHI:
    """
    卡方检验
    """

    def __init__(self, train_df=None):

        # self.train_words_file = 'train_words.csv'
        self.train_words_file = 'train_tags.csv'
        self.chi_file = 'chi.txt'
        self.chi = {}
        self.pos = {}
        self.neg = {}

        self.train_df = train_df
        if not train_df:
            self.train_df = load_csv(self.train_words_file)

    def compute_chi(self):
        # 统计样本数
        total = self.train_df.shape[0]
        total_pos = self.train_df[self.train_df['label'] == 'POSITIVE'].shape[0]
        total_neg = self.train_df[self.train_df['label'] == 'NEGATIVE'].shape[0]
        print('总数:', total, '正样本数', total_pos, '负样本数', total_neg)
        # 初始化 chi
        for w in self.pos:
            self.chi[w] = 0
            if w not in self.neg:
                self.neg[w] = 0  # 标记负样本中没有该词
        for w in self.neg:
            if w not in self.chi:
                self.chi[w] = 0
                if w not in self.pos:
                    self.pos[w] = 0  # 标记正样本中没有该词
        # 计算 chi
        for w in self.chi:
            A = self.pos[w]
            C = total_pos - self.pos[w]
            B = self.neg[w]
            D = total_neg - self.neg[w]
            self.chi[w] = (A * D - B * C) ** 2 / ((A + B) * (C + D))
        # 排序 降序
        sort = sorted(self.chi.items(), key=lambda e: e[1], reverse=True)
        # 写文件
        fw = codecs.open(self.chi_file, 'w', encoding='utf-8')
        for item in sort:
            fw.write('%s %s\n' % (item[0], math.log10(item[1] + 1)))

    def pos_count(self):
        train_df = self.train_df[self.train_df['label'] == 'POSITIVE']
        for n in range(train_df.shape[0]):  # 遍历行
            words = []
            try:
                words.extend(train_df.iloc[n]['head'].split())
            except Exception:
                print('%s head is nan' % n)
            try:
                words.extend(train_df.iloc[n]['content'].split())
            except Exception:
                print('%s content is nan' % n)
            done = {}  # 该行已经统计过的词
            for w in words:  # 遍历词
                if w not in done:  # n 行没有统计过 w 词
                    done[w] = True
                    if w not in self.pos:
                        self.pos[w] = 1
                    else:
                        self.pos[w] += 1
                else:
                    continue
            # 跟踪进度
            if (n + 1) % 10000 == 0:
                time_str = datetime.datetime.now().isoformat()
                print('%s, %s line done' % (time_str, n+1))

    def neg_count(self):
        train_df = self.train_df[self.train_df['label'] == 'NEGATIVE']
        for n in range(train_df.shape[0]):  # 遍历行
            words = []
            try:
                words.extend(train_df.iloc[n]['head'].split())
            except:
                print('%s head is nan' % n)
            try:
                words.extend(train_df.iloc[n]['content'].split())
            except:
                print('%s content is nan' % n)
            done = {}  # 该行已经统计过的词
            for w in words:  # 遍历词
                if w not in done:  # n 行没有统计过 w 词
                    done[w] = True
                    if w not in self.neg:
                        self.neg[w] = 1
                    else:
                        self.neg[w] += 1
                else:
                    continue
            # 跟踪进度
            if (n + 1) % 10000 == 0:
                time_str = datetime.datetime.now().isoformat()
                print('%s, %s done' % (time_str, n+1))

    def run(self):
        ts = [threading.Thread(target=self.pos_count, args=()),
              threading.Thread(target=self.neg_count, args=())]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        self.compute_chi()


class TFIDF:

    def __init__(self):

        self.train_tags_pos_file = 'train_tags_pos.csv'
        self.train_tags_neg_file = 'train_tags_neg.csv'
        self.train_tags_file = 'train_tags.csv'
        self.test_tags_file = 'test_tags.csv'

        self.stop_words = {}
        self.idf = {}

    def set_idf_file(self, idf_file):
        with open(idf_file, 'r') as f:
            for line in f:
                line = line.strip().split()
                self.idf[line[0]] = float(line[1])

    def set_stop_words(self, stop_words_file):
        with open(stop_words_file, 'r') as f:
            for line in f:
                self.stop_words[line.strip()] = True

    def extract_tags(self, words, topK=20, withWeight=False):
        freq = {}
        for w in words:
            if len(w.strip()) < 2 or w in self.stop_words:
                continue
            freq[w] = freq.get(w, 0.0) + 1.0
        total = sum(freq.values())  # 总词数
        for k in freq:  # 遍历词典的 key
            freq[k] = freq[k] * self.idf[k] / total
        if withWeight:  # 如果要返回权重
            tags = sorted(freq.items(), key=itemgetter(1), reverse=True)
            return tags[:topK]
        # 如果不需要返回权重
        rtags = []
        tags = sorted(freq, key=freq.__getitem__, reverse=True)
        tags = tuple(tags[:topK])
        for w in words:
            if w in tags and w not in rtags:  # 保证一个词只返回一次
                rtags.append(w)
        return rtags

    def extract_train_tags(self, train_df=None, head_topK=6, content_topK=50, TF=False, withWeight=False):
        """
        提取关键词
        :param train_df: 训练集数据框
        :param head_topK: 要从标题中提取的关键词数
        :param content_topK: 要从内容中提取的关键词数
        """
        if TF:
            for w in self.idf:
                self.idf[w] = 1
        if not train_df:
            train_df = load_csv('train_words.csv')
        fw_pos = codecs.open(self.train_tags_pos_file, 'w', encoding='utf-8')
        fw_neg = codecs.open(self.train_tags_neg_file, 'w', encoding='utf-8')
        for n in range(train_df.shape[0]):
            tags_head = []
            tags_content = []
            try:
                tags_head = self.extract_tags(train_df.iloc[n]['head'].split(),
                                              topK=head_topK,
                                              withWeight=withWeight)
            except:
                print('%s head is nan' % n)
            while len(tags_head) < head_topK: tags_head.append('<PAD_HEAD>')
            try:
                tags_content = self.extract_tags(train_df.iloc[n]['content'].split(),
                                                 topK=content_topK,
                                                 withWeight=withWeight)
            except:
                print('%s content is nan' % n)
            while len(tags_content) < content_topK: tags_content.append('<PAD_CONTENT>')
            tags = tags_head + tags_content

            if train_df.iloc[n]['label'] == 'POSITIVE':
                fw_pos.write(' '.join(tags) + '\n')
            else:
                fw_neg.write(' '.join(tags) + '\n')
        fw_pos.close()
        fw_neg.close()
        print('extract train tags done')

    def extract_tags_all(self, df_list=None, head_topK=6, content_topK=100, TF=False, withWeight=False):
        if not df_list:
            # df_list = [load_csv('train_words.csv'), load_csv('test_words.csv')]
            df_list = [load_csv('train_words.csv')]
        for df in df_list:
            if TF:
                for w in self.idf:
                    self.idf[w] = 1
            fw = codecs.open(self.train_tags_file, 'w', encoding='utf-8')
            for n in range(df.shape[0]):
                tags_head = []
                tags_content = []
                try:
                    tags_head = self.extract_tags(df.iloc[n]['head'].split(),
                                                  topK=head_topK,
                                                  withWeight=withWeight)
                except:
                    print('%s head is nan' % n)
                while len(tags_head) < head_topK: tags_head.append('<PAD_HEAD>')
                try:
                    tags_content = self.extract_tags(df.iloc[n]['content'].split(),
                                                     topK=content_topK,
                                                     withWeight=withWeight)
                except:
                    print('%s content is nan' % n)
                while len(tags_content) < content_topK: tags_content.append('<PAD_CONTENT>')

                fw.write('%s\t%s\t%s\t%s\n' % ( df.iloc[n]['id'],
                                                ' '.join(tags_head),
                                                ' '.join(tags_content),
                                                df.iloc[n]['label']))
            fw.close()

