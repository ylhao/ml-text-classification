# coding: utf-8

import cfg
import codecs
from nlp import load_csv


class IDF:

    def __init__(self, df_list=None):

        self.idf_file = cfg.DATA_PATH + 'idf.txt'  # idf文档
        self.idf = {}  # 统计该词的 idf

        self.df_list = df_list
        if not df_list:
            self.df_list = [load_csv(cfg.DATA_PATH + 'train_words.csv'), load_csv(cfg.DATA_PATH + 'test_words.csv')]

    def compute_idf(self):
        total_num = 0  # 文章总数
        for df in self.df_list:
            total_num += df.shape[0]
            for n in range(df.shape[0]):
                done = {}
                words = []
                try:
                    words.extend(df.iloc[n]['head'].split())
                except Exception:
                    print('%s head is nan' % n)
                try:
                    words.extend(df.iloc[n]['content'].split())
                except Exception:
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
            self.idf[w] = float(total_num) / self.idf[w]
        with codecs.open(self.idf_file, 'w', encoding='utf-8') as f:
            for w in self.idf:
                if self.idf[w] != 0:
                    f.write('%s %f\n' % (w, self.idf[w]))
        print('save %s done' % self.idf_file)

    def run(self):
        self.compute_idf()