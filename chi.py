import cfg
import codecs
import threading
import datetime
from nlp import load_csv


class CHI:

    def __init__(self, train_df=None):

        self.chi_file = cfg.DATA_PATH + 'chi.txt'  # chi文档
        self.chi = {}
        self.pos = {}
        self.neg = {}

        self.train_df = train_df
        if not train_df:
            self.train_df = load_csv(cfg.DATA_PATH + 'train_words.csv')

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
        # 计算 CHI
        for w in self.chi:
            A = self.pos[w]
            C = total_pos - self.pos[w]
            B = self.neg[w]
            D = total_neg - self.neg[w]
            self.chi[w] = (A * D - B * C) ** 2 / ((A + B) * (C + D))
        # 排序 降序
        tags = sorted(self.chi, key=self.chi.__getitem__, reverse=True)
        # 写文件
        fw = codecs.open(self.chi_file, 'w', encoding='utf-8')
        for w in tags:
            fw.write('%s\n' % w)

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
